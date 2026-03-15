// Command bench-note attaches benchmark results to git commits as notes
// using txtar format for structured storage.
//
// Usage:
//
//	bench-note run [flags]         Run benchmarks and attach as git note to HEAD
//	bench-note show [commit]       Display bench note for a commit
//	bench-note raw [commit]        Extract raw go test output (for benchstat input)
//	bench-note compare c1 c2      Run benchstat between two commits' bench notes
//	bench-note history [--oneline] List commits that have bench notes
package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"golang.org/x/tools/txtar"
)

const notesRef = "refs/notes/benchmarks"

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd := os.Args[1]
	var err error
	switch cmd {
	case "run":
		err = cmdRun(os.Args[2:])
	case "show":
		err = cmdShow(os.Args[2:])
	case "raw":
		err = cmdRaw(os.Args[2:])
	case "compare":
		err = cmdCompare(os.Args[2:])
	case "history":
		err = cmdHistory(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "bench-note %s: %v\n", cmd, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintf(os.Stderr, `Usage: bench-note <command> [flags]

Commands:
  run [--benchtime=5x] [--count=6] [--from-file=FILE]
      Run benchmarks and attach as git note to HEAD.
  show [commit]
      Display bench note (default: HEAD).
  raw [commit]
      Extract raw go test output for benchstat input.
  compare <commit1> <commit2>
      Run benchstat between two commits.
  history [--oneline]
      List commits with bench notes.
`)
	os.Exit(2)
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	benchtime := fs.String("benchtime", "5x", "benchtime flag for go test")
	count := fs.Int("count", 6, "number of benchmark runs")
	fromFile := fs.String("from-file", "", "use existing benchmark output file instead of running")
	fs.Parse(args)

	commit, err := gitRevParse("HEAD")
	if err != nil {
		return err
	}

	var raw []byte
	var exitCode int
	if *fromFile != "" {
		raw, err = os.ReadFile(*fromFile)
		if err != nil {
			return fmt.Errorf("reading file: %w", err)
		}
	} else {
		raw, exitCode, err = runBenchmarks(*benchtime, *count)
		if err != nil {
			return fmt.Errorf("running benchmarks: %w", err)
		}
	}

	goVersion, _ := gitOutput("go", "version")
	goVersion = strings.TrimSpace(goVersion)

	// Build metadata comment.
	var comment strings.Builder
	fmt.Fprintf(&comment, "bench-note v1\n")
	fmt.Fprintf(&comment, "commit: %s\n", commit[:minInt(len(commit), 12)])
	fmt.Fprintf(&comment, "date: %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintf(&comment, "go-version: %s\n", goVersion)
	fmt.Fprintf(&comment, "benchtime: %s\n", *benchtime)
	fmt.Fprintf(&comment, "count: %d\n", *count)
	fmt.Fprintf(&comment, "exit-code: %d\n", exitCode)

	// Build txtar archive.
	ar := &txtar.Archive{
		Comment: []byte(comment.String()),
		Files: []txtar.File{
			{Name: "raw.txt", Data: raw},
		},
	}

	// Find parent with bench note and run benchstat.
	parentCommit, parentRaw, err := findParentBenchNote(commit)
	if err == nil && len(parentRaw) > 0 {
		delta, err := runBenchstat(parentRaw, raw)
		if err == nil {
			ar.Files = append(ar.Files, txtar.File{
				Name: fmt.Sprintf("benchstat-vs-%s.txt", parentCommit[:minInt(len(parentCommit), 8)]),
				Data: delta,
			})
		}
	}

	note := txtar.Format(ar)

	// Attach as git note.
	if err := gitNoteAdd(commit, note); err != nil {
		return fmt.Errorf("adding note: %w", err)
	}
	fmt.Fprintf(os.Stderr, "bench note attached to %s\n", commit[:minInt(len(commit), 8)])
	return nil
}

func cmdShow(args []string) error {
	commit := "HEAD"
	if len(args) > 0 {
		commit = args[0]
	}
	note, err := readNote(commit)
	if err != nil {
		return err
	}
	os.Stdout.Write(note)
	return nil
}

func cmdRaw(args []string) error {
	commit := "HEAD"
	if len(args) > 0 {
		commit = args[0]
	}
	note, err := readNote(commit)
	if err != nil {
		return err
	}
	ar := txtar.Parse(note)
	for _, f := range ar.Files {
		if f.Name == "raw.txt" {
			os.Stdout.Write(f.Data)
			return nil
		}
	}
	return fmt.Errorf("no raw.txt in bench note")
}

func cmdCompare(args []string) error {
	if len(args) < 2 {
		return fmt.Errorf("usage: bench-note compare <commit1> <commit2>")
	}
	raw1, err := extractRaw(args[0])
	if err != nil {
		return fmt.Errorf("commit %s: %w", args[0], err)
	}
	raw2, err := extractRaw(args[1])
	if err != nil {
		return fmt.Errorf("commit %s: %w", args[1], err)
	}
	delta, err := runBenchstat(raw1, raw2)
	if err != nil {
		return err
	}
	os.Stdout.Write(delta)
	return nil
}

func cmdHistory(args []string) error {
	fs := flag.NewFlagSet("history", flag.ExitOnError)
	oneline := fs.Bool("oneline", false, "compact output")
	fs.Parse(args)

	// List all notes in refs/notes/benchmarks.
	out, err := gitOutput("git", "notes", "--ref="+notesRef, "list")
	if err != nil {
		return fmt.Errorf("no bench notes found")
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for _, line := range lines {
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}
		commitHash := parts[1]
		if *oneline {
			subject, _ := gitOutput("git", "log", "-1", "--format=%h %s", commitHash)
			fmt.Print(strings.TrimSpace(subject), "\n")
		} else {
			subject, _ := gitOutput("git", "log", "-1", "--format=%h %s (%ar)", commitHash)
			fmt.Println(strings.TrimSpace(subject))
			// Show first few lines of the note comment.
			note, err := readNoteForHash(commitHash)
			if err == nil {
				ar := txtar.Parse(note)
				commentLines := strings.Split(string(ar.Comment), "\n")
				for _, cl := range commentLines {
					if cl == "" {
						continue
					}
					fmt.Printf("  %s\n", cl)
				}
			}
			fmt.Println()
		}
	}
	return nil
}

// runBenchmarks runs go test -bench and returns the output.
func runBenchmarks(benchtime string, count int) ([]byte, int, error) {
	args := []string{"test", "-bench=.", "-benchmem",
		fmt.Sprintf("-benchtime=%s", benchtime),
		fmt.Sprintf("-count=%d", count),
		"-run=^$", "-timeout=30m",
	}
	fmt.Fprintf(os.Stderr, "running: go %s\n", strings.Join(args, " "))
	cmd := exec.Command("go", args...)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	err := cmd.Run()
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			return nil, 0, err
		}
	}
	return buf.Bytes(), exitCode, nil
}

// findParentBenchNote walks first-parent history to find the nearest
// ancestor with a bench note.
func findParentBenchNote(commit string) (string, []byte, error) {
	out, err := gitOutput("git", "log", "--first-parent", "--format=%H", commit)
	if err != nil {
		return "", nil, err
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	// Skip the first line (current commit).
	for _, hash := range lines[1:] {
		hash = strings.TrimSpace(hash)
		if hash == "" {
			continue
		}
		raw, err := extractRawForHash(hash)
		if err == nil {
			return hash, raw, nil
		}
	}
	return "", nil, fmt.Errorf("no parent with bench note")
}

func extractRaw(ref string) ([]byte, error) {
	note, err := readNote(ref)
	if err != nil {
		return nil, err
	}
	ar := txtar.Parse(note)
	for _, f := range ar.Files {
		if f.Name == "raw.txt" {
			return f.Data, nil
		}
	}
	return nil, fmt.Errorf("no raw.txt in bench note")
}

func extractRawForHash(hash string) ([]byte, error) {
	note, err := readNoteForHash(hash)
	if err != nil {
		return nil, err
	}
	ar := txtar.Parse(note)
	for _, f := range ar.Files {
		if f.Name == "raw.txt" {
			return f.Data, nil
		}
	}
	return nil, fmt.Errorf("no raw.txt in bench note")
}

func readNote(ref string) ([]byte, error) {
	hash, err := gitRevParse(ref)
	if err != nil {
		return nil, err
	}
	return readNoteForHash(hash)
}

func readNoteForHash(hash string) ([]byte, error) {
	out, err := gitOutput("git", "notes", "--ref="+notesRef, "show", hash)
	if err != nil {
		return nil, fmt.Errorf("no bench note for %s", hash[:minInt(len(hash), 8)])
	}
	return []byte(out), nil
}

func gitNoteAdd(commit string, content []byte) error {
	cmd := exec.Command("git", "notes", "--ref="+notesRef, "add", "-f", "-m", string(content), commit)
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func gitRevParse(ref string) (string, error) {
	out, err := gitOutput("git", "rev-parse", ref)
	if err != nil {
		return "", fmt.Errorf("rev-parse %s: %w", ref, err)
	}
	return strings.TrimSpace(out), nil
}

func gitOutput(name string, args ...string) (string, error) {
	cmd := exec.Command(name, args...)
	out, err := cmd.Output()
	return string(out), err
}

func runBenchstat(old, new []byte) ([]byte, error) {
	oldFile, err := os.CreateTemp("", "bench-old-*.txt")
	if err != nil {
		return nil, err
	}
	defer os.Remove(oldFile.Name())

	newFile, err := os.CreateTemp("", "bench-new-*.txt")
	if err != nil {
		return nil, err
	}
	defer os.Remove(newFile.Name())

	if _, err := oldFile.Write(old); err != nil {
		return nil, err
	}
	oldFile.Close()

	if _, err := newFile.Write(new); err != nil {
		return nil, err
	}
	newFile.Close()

	cmd := exec.Command("benchstat", oldFile.Name(), newFile.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("benchstat: %w\n%s", err, out)
	}
	return out, nil
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
