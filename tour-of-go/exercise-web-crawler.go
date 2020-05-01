package main

import (
	"fmt"
	"sync"
	"time"
)

type Fetcher interface {
	// Fetch returns the body of URL and
	// a slice of URLs found on that page.
	Fetch(url string) (body string, urls []string, err error)
}

// Crawl uses fetcher to recursively crawl
// pages starting with url, to a maximum of depth
func Crawl(url string, depth int, fetcher Fetcher) {
	// This implementation doesn't do either
	cache := make(map[string]int)
	var mutex sync.Mutex
	var crawl_sub func(string, int)
	crawl_sub = func(url string, depth int) {
		if depth <= 0 {
			fmt.Printf("too deep: %s\n", url)
			return
		}
		// Don't fetch the same URL twice.
		mutex.Lock()
		cache[url]++
		c := cache[url]
		mutex.Unlock()
		if c > 1 {
			fmt.Printf("fetched: %s\n", url)
			return
		}
		// Fetch URLs in parallel
		body, urls, err := fetcher.Fetch(url)
		if err != nil {
			fmt.Println(err)
			return
		}
		fmt.Printf("found: %s\n", url, body)
		for _, u := range urls {
			go crawl_sub(u, depth-1)
		}
	}

	crawl_sub(url, depth)
	time.Sleep(time.Second)
	return
}

func main() {
	Crawl("https://golang.org/", 4, fetcher)
}

type fakeFetcher map[string]*fakeResult

type fakeResult struct {
	body string
	urls []string
}

func (f fakeFetcher) Fetch(url string) (string, []string, error) {
	if res, ok := f[url]; ok {
		return res.body, res.urls, nil
	}
	return "", nil, fmt.Errorf("not found: %s", url)
}

// fetcher is a populated fakeFetcher.
var fetcher = fakeFetcher{
	"https://golang.org/": &fakeResult{
		"The Go Programming Language",
		[]string{
			"https://golang.org/pkg/",
			"https://golang.org/cmd/",
		},
	},
	"https://golang.org/pkg/": &fakeResult{
		"Packages",
		[]string{
			"https://golang.org/",
			"https://golang.org/cmd/",
			"https://golang.org/pkg/fmt/",
			"https://golang.org/pkg/os/",
		},
	},
	"https://golang.org/pkg/fmt/": &fakeResult{
		"Package fmt",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
	"https://golang.org/pkg/os/": &fakeResult{
		"Package os",
		[]string{
			"https://golang.org/",
			"https://golang.org/pkg/",
		},
	},
}
