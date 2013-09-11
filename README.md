# rust-ml

Machine learning library for the Rust programming language.

Features:
* clustering
  * k-means
* regression
  * linear regression

To build rust-ml:

	$ make

To build and run examples: (You will need to have gnuplot installed to run them).

	$ make examples
	$ ./out/kmeans
	$ ./out/linalg
