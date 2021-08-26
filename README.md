# Image-Inpainting-via-DL

Graduation project at UCAS

Abstract: Image is the most important information carrier in our lives. It is an objective, vivid, specific, and intuitive description of the characteristics of things. Therefore, finding efficient image inpainting methods has been a spotlight in research. With the development and prevalence of computer technology, digital image inpainting has gradually replaced manual restoration as the mainstream method of image inpainting. Sparse-dictionary-learning based image inpainting approaches assume that the image can be sparsely represented under a certain basis, so the original image can be restored through dictionary learning, which learns the basis from the known information, and sparse coding, which finds the sparse representation of the image under the learned basis. 

This thesis studies sparse-dictionary-learning based image inpainting approaches, including the following main ingredients: 
Discusses the mathematical theory of sparse coding and dictionary learning with highlights on several representative optimization algorithms. Surveys the research progress of dictionary learning, including both dictionary learning applications and theories.
Proposes using dual-patch extraction to convert the image to the input data. Uses a class of block coordinate descent algorithms to solve a specific optimization model of image inpainting, and establishes the global convergence of the algorithm based on existing literature.
Designs and carries out a series of numerical simulations to test the performance of the algorithm in real applications, and compares the algorithm with the commonly used K-SVD image inpainting method. Discusses and interprets the experimental results.

The results of numerical experiments show that the main advantage of the method in this thesis is the guarantee of global convergence, while this method does not enjoy obvious superiority over traditional methods in the inpainting of real images.
