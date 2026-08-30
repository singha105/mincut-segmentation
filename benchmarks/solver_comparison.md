| Image | Cut grid | Nodes | Edmonds-Karp (from scratch) | Boykov-Kolmogorov (PyMaxflow) | Speed-up | Same result |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| dog.jpg | 56x56 | 3,136 | 0.87s | 0.001s | 952x | yes |
| hen.jpg | 40x56 | 2,240 | 1.20s | 0.001s | 1216x | yes |
| horse.jpg | 56x55 | 3,080 | 1.03s | 0.001s | 1220x | yes |
| tiger.jpg | 37x56 | 2,072 | 0.64s | 0.000s | 1348x | yes |
| dog.jpg | 96x96 | 9,216 | 2.49s | 0.002s | 1376x | yes |
| hen.jpg | 69x96 | 6,624 | 9.70s | 0.005s | 2139x | yes |
| horse.jpg | 96x95 | 9,120 | 2.70s | 0.002s | 1358x | yes |
| tiger.jpg | 64x96 | 6,144 | 4.12s | 0.002s | 1785x | yes |
| dog.jpg | 160x160 | 25,600 | not run | 0.007s | n/a | n/a |
| hen.jpg | 114x160 | 18,240 | not run | 0.013s | n/a | n/a |
| horse.jpg | 160x158 | 25,280 | not run | 0.006s | n/a | n/a |
| tiger.jpg | 106x160 | 16,960 | not run | 0.007s | n/a | n/a |

End-to-end `segment_image()` on 800px inputs, worst of the four photos:

| Mode | Cut grid cap | Slowest photo |
| --- | ---: | ---: |
| Edmonds-Karp (default) | 56px | 1.26s |
| Boykov-Kolmogorov | 56px | 0.01s |
| Boykov-Kolmogorov | 256px | 0.04s |
| Boykov-Kolmogorov | 400px | 0.08s |
