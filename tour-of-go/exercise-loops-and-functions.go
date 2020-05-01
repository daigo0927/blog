package main

import (
	"fmt"
	"math"
)

func Sqrt(x float64) (int, float64) {
	var z float64 = 1
	for i := 0; i < 10; i++ {
		z_new := z - (z*z-x)/(2*z)
		if math.Abs(z_new-z) < 1e-10 {
			return i, z
		}
		z = z_new
	}
	return 10, z
}

func main() {
	fmt.Println(Sqrt(2))
	fmt.Println(math.Sqrt(2))
}
