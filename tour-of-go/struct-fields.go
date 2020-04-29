package main

import "fmt"

type Version struct {
	X int
	Y int
}

func main() {
	v := Version{X: 1, Y: 2}
	v.X = 4
	fmt.Println(v.X)
}
