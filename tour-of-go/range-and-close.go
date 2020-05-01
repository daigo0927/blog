package main

import "fmt"

func fibonacci(n int, c chan int) {
	x, y := 0, 1
	for int := 0; int < n; int++ {
		c <- x
		x, y = y, x + y
	}
	close(c)
}

func main() {
	c := make(chan int, 10)
	go fibonacci(cap(c), c)
	for i := range c { // iterator-like usage in python
		fmt.Println(i)
	}
}
