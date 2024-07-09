<span style="font-family: 'Times New Roman'; font-size:20px">

<h1 align="center">
CS-328 Data Science Project

Sampling and Sketching Methods for Novel Matrix Factorization
</h1>

<h2>Some Important Terms you may need to know:</h2>
<h3>
What is Matrix Factorization?
</h3>

We have a huge matrix $A$ of size $m \times n$, we want to store this matrix but as it is of huge size, it is not possible to store it in memory. So, we use matrix factorization techniques to store this matrix $A$ by applying some Sampling and Sketching Techniques on the matrix and then by factorizing it into two matrices $U$ and $V'$ of size $m \times k$ and $k \times n$ respectively. We can store these two matrices in memory and can use them to get the original matrix $A$ by multiplying these two matrices $U$ and $V'$. We have to find the best $k$ value for which we can get the original matrix $A$ with minimum error.

$$A = U \times V\ |\ A → m \times n,\ \ \ U → m \times k,\ \ \ V → k \times n$$

But after applying some Sampling and Sketching, we get $U$ and $V'$ instead of $V$, and $A$ can be approximated as:

$$A' = U \times V'\ |\ A \approx A'$$

<h3>
What is Sketching and Sampling?
</h3>

**Sampling:** Sampling is the process of selecting a subset of the data from the original data set. It is used to reduce the size of the data set because we can't store the huge data set in our local devices. So, we need to sample the data set to reduce its size. Sampling can be done in two ways:
- Random Sampling
- Stratified Sampling
- Cluster Sampling
- Systematic Sampling
- Convenience Sampling
- Snowball Sampling
- Quota Sampling
- Judgmental Sampling
- Purposive Sampling

</span>
