module github.com/tmc/autoresearch-go-ane

go 1.25.2

require github.com/tmc/apple v0.3.0

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/tmc/aneperf v0.0.0
)

replace github.com/ebitengine/purego => github.com/tmc/purego v0.10.0-alpha.2.0.20260130081008-0b23e28544a2

replace github.com/tmc/aneperf => /Users/tmc/go/src/github.com/tmc/aneperf
