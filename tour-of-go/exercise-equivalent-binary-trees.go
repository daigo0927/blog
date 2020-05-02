package main

import (
	"fmt"
	"golang.org/x/tour/tree"
)

// // Tree structure
// type tree struct {
// 	Left *tree
// 	Value int
// 	Right *tree
// }

// Walk walks the tree t sending all values
// from the tree to the channel ch.
func Walk(t *tree.Tree, ch chan int) {
	var walk_sub func(*tree.Tree)
	walk_sub = func(t *tree.Tree) {
		if t != nil {
			walk_sub(t.Left)
			ch <- t.Value
			walk_sub(t.Right)
		}
	}
	walk_sub(t)
	close(ch)
}

// Same determines whether the trees
// t1 and t2 contain the same values
func Same(t1, t2 *tree.Tree) bool {
	ch1, ch2 := make(chan int), make(chan int)
	go Walk(t1, ch1)
	go Walk(t2, ch2)
	for v1 := range ch1 {
		v2 := <-ch2
		if v1 != v2 {
			return false
		}
	}
	return true
}

func main() {
	ch := make(chan int)
	go Walk(tree.New(1), ch)

	for v := range ch {
		fmt.Println(v)
	}

	var comp bool
	comp = Same(tree.New(1), tree.New(1))
	fmt.Printf("Comparison 1vs1: %v\n", comp)

	comp = Same(tree.New(1), tree.New(2))
	fmt.Printf("Comparison 1vs2: %v\n", comp)
}
