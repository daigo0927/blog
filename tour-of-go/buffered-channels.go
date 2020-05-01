package main

import "fmt"

func main() {
	ch := make(chan int, 2) // channel with size 2 buffer
	ch <- 1
	ch <- 2
	// ch <- 3 // over buffer
	fmt.Println(<-ch)
	fmt.Println(<-ch)
	// fmt.Println(<-ch) // empty buffer
	fmt.Printf("Remaining buffer: %v\n", len(ch)) // remaining buffer size
}
