//go:build !darwin

package main

import "fmt"

func main() {
	fmt.Println("autoresearch-go-ane requires macOS with Apple Silicon")
}
