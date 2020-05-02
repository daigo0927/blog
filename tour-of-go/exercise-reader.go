package main

import "golang.org/x/tour/reader"

type MyReader struct{}

func (r MyReader) Read(b []byte) (int, error) {
	var i int
	var e error
	for i, e = 0, nil; i < len(b); i++ {
		b[i] = 'A'
	}
	return i, e
}

func main() {
	reader.Validate(MyReader{})
}
