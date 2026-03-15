module github.com/tmc/autoresearch-go-ane

go 1.25.2

require github.com/tmc/apple v0.3.0

require (
	github.com/ebitengine/purego v0.10.0 // indirect
	github.com/tmc/aneperf v0.0.0
)

replace github.com/ebitengine/purego => github.com/tmc/purego v0.10.0-alpha.2.0.20260315005611-7a4926ded181

replace github.com/tmc/aneperf => /Users/tmc/go/src/github.com/tmc/aneperf
