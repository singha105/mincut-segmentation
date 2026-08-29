| Image | Cut grid | Nodes | Edmonds-Karp (from scratch) | Boykov-Kolmogorov (PyMaxflow) | Speed-up | Same result |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| dog.jpg | 56x56 | 3,136 | 0.86s | 0.001s | 1116x | yes |
| hen.jpg | 40x56 | 2,240 | 1.27s | 0.001s | 1312x | yes |
| horse.jpg | 56x55 | 3,080 | 1.05s | 0.001s | 1028x | yes |
| tiger.jpg | 37x56 | 2,072 | 0.65s | 0.000s | 1332x | yes |
| dog.jpg | 96x96 | 9,216 | 2.53s | 0.002s | 1337x | yes |
| hen.jpg | 69x96 | 6,624 | 9.66s | 0.005s | 2038x | yes |
| horse.jpg | 96x95 | 9,120 | 2.72s | 0.002s | 1333x | yes |
| tiger.jpg | 64x96 | 6,144 | 4.12s | 0.002s | 1812x | yes |
| dog.jpg | 160x160 | 25,600 | not run | 0.007s | — | — |
| hen.jpg | 114x160 | 18,240 | not run | 0.013s | — | — |
| horse.jpg | 160x158 | 25,280 | not run | 0.007s | — | — |
| tiger.jpg | 106x160 | 16,960 | not run | 0.007s | — | — |

End-to-end `segment_image()` on 800px inputs, worst of the four photos:

| Mode | Cut grid cap | Slowest photo |
| --- | ---: | ---: |
| Edmonds-Karp (default) | 56px | 1.23s |
| Boykov-Kolmogorov | 56px | 0.01s |
| Boykov-Kolmogorov | 256px | 0.05s |
| Boykov-Kolmogorov | 400px | 0.08s |
