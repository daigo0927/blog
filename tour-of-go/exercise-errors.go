package main

import (
	"fmt"
	"math"
)

func Sqrt(x float64) (float64, error) {
	if x < 0 {
		return 0, ErrNegativeSqrt(x)
	} else {
		return math.Sqrt(x), nil
	}
}

type ErrNegativeSqrt float64

func (e ErrNegativeSqrt) Error() string {
	v := float64(e)
	return fmt.Sprintf("cannot Sqrt negative number: %v", v)
}

func main() {
	fmt.Println(Sqrt(2))
	fmt.Println(Sqrt(-2))
}
