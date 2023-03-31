"""
Optimization related functions. Needed by those quantifiers that solve optimization problems to compute the
estimated prevalence given a testing bag
"""

# Authors: Alberto Castaño <bertocast@gmail.com>
#          Pablo González <gonzalezgpablo@uniovi.es>
#          Jaime Alonso <jalonso@uniovi.es>
#          Juan José del Coz <juanjo@uniovi.es>
# License: GPLv3 3 clause, University of Oviedo

import numpy as np
import cvxpy
import quadprog
import scipy


############
#  Functions for solving optimization problems using CVXPY, loss functions: L1, HD
############
def solve_l1(train_distrib, test_distrib, n_classes, problem=None, solver=None):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L1 loss function

        min   |train_distrib * prevalences - test_distrib|
        s.t.  prevalences_i >=0
              sum prevalences_i = 1

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC, Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        n_classes : int
            Number of classes

        problem : a cvxpy Problem object (default=None)
            The first time a problem is solved (this corresponds to the first testing bag) this parameter is None. For
            the rest testing bags, a Problem object is passed here to allow a warm start. This accelerates the solving
            process.

        solver : str, (default=None)
            The solver used to solve the optimization problem. cvxpy automatically selects the best solver (among those
            installed) according to the type of the optimization problem. If a particular solver is prefered maybe
            you need to install additional libraries

        Returns
        -------
        problem : a cvxpy Problem
            A cvxpy Problem already created

        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    if problem is None:
        prevalences = cvxpy.Variable(n_classes, nonneg=True)
        test_d = cvxpy.Parameter(len(test_distrib), nonneg=True)
        objective = cvxpy.Minimize(cvxpy.norm(test_d - train_distrib @ prevalences, 1))
        test_d.value = np.squeeze(test_distrib)
        contraints = [cvxpy.sum(prevalences) == 1]

        problem = cvxpy.Problem(objective, contraints)
        if solver is None:
            problem.solve()
        else:
            problem.solve(solver=solver)

        return problem, np.array(prevalences[0:n_classes].value).squeeze()
    else:
        problem.parameters()[0].value = np.squeeze(test_distrib)
        if solver is None:
            problem.solve(warm_start=True)
        else:
            problem.solve(warm_start=True, solver=solver)

        return problem, list(problem.solution.primal_vars.values())[0]


def solve_hd(train_distrib, test_distrib, n_classes, problem=None, solver='ECOS'):
    """ Solves the optimization problem for PDF methods using Hellinger Distance

        This method just uses cvxpy library

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        test_distrib : array, shape (n_bins * n_classes, 1)
            Represents the distribution of the testing set

        n_classes : int
            Number of classes

        problem : a cvxpy Problem object (default=None)
            The first time a problem is solved (this corresponds to the first testing bag) this parameter is None. For
            the rest testing bags, a Problem object is passed here to allow a warm start. This accelerates the solving
            process.

        solver : str, (default='ECOS')
            The solver used to solve the optimization problem. Here 'ECOS' is used. If another solver is prefered,
            you may need to install additional libraries

        Returns
        -------
        prevalences : array, shape=(n_classes, )
            Vector containing the predicted prevalence for each class
    """
    if problem is None:
        prevalences = cvxpy.Variable(n_classes, nonneg=True)
        test_d = cvxpy.Parameter(len(test_distrib), nonneg=True)
        s = cvxpy.multiply(test_d, train_distrib @ prevalences)
        test_d.value = np.squeeze(test_distrib)
        objective = cvxpy.Minimize(1 - cvxpy.sum(cvxpy.sqrt(s)))
        contraints = [cvxpy.sum(prevalences) == 1]
        problem = cvxpy.Problem(objective, contraints)
        problem.solve(solver=solver, feastol=1e6)

        return problem, np.array(prevalences[0:n_classes].value).squeeze()
    else:
        problem.parameters()[0].value = np.squeeze(test_distrib)
        problem.solve(warm_start=True, solver=solver, feastol=1e6)

        return problem, list(problem.solution.primal_vars.values())[0]


############
#  Functions for solving optimization problems using QUADPROG (L2 loss function)
############
def solve_l2(train_distrib, test_distrib, G, C, b):
    """ Solves AC, PAC, PDF and Friedman optimization problems for L2 loss function

        min    (test_distrib - train_distrib * prevalences).T (test_distrib - train_distrib * prevalences)
        s.t.   prevalences_i >=0
               sum prevalences_i = 1

        Expanding the objective function, we obtain:

        prevalences.T train_distrib.T train_distrib prevalences
        - 2 prevalences train_distrib.T test_distrib + test_distrib.T test_distrib

        Notice that the last term is constant w.r.t prevalences.

        Let G = 2 train_distrib.T train_distrib  and a = 2 * train_distrib.T test_distrib, we can use directly
        quadprog.solve_qp because it solves the following kind of problems:

        Minimize     1/2 x^T G x - a^T x
        Subject to   C.T x >= b

        `solve_l2` just computes the term a, shape (n_classes,1), and then calls quadprog.solve_qp.
        G, C and b were computed by `compute_l2_param_train` before, in the 'fit' method` of the PDF/Friedman object.
        quadprog is used here because it is faster than cvxpy.

        Parameters
        ----------
        train_distrib : array, shape depends on the optimization problem
            Represents the distribution of each class in the training set
            PDF: shape (n_bins * n_classes, n_classes)
            AC, PAC Friedman: shape (n_classes, n_classes)

        test_distrib : array, shape depends on the optimization problem
            Represents the distribution of the testing set
            PDF: shape shape (n_bins * n_classes, 1)
            AC, PAC, Friedman: shape (n_classes, 1)

        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1

        b : array, shape (n_constraints,)

        G, C and b are computed by `compute_l2_param_train` in the 'fit' method

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class
    """
    a = 2 * train_distrib.T.dot(test_distrib)
    a = np.squeeze(a)
    prevalences = quadprog.solve_qp(G=G, a=a, C=C, b=b, meq=1)
    return prevalences[0]


def compute_l2_param_train(train_distrib, classes):
    """ Computes params related to the train distribution for solving PDF optimization problems using
        L2 loss function

        Parameters
        ----------
        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        Returns
        -------
        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1  (n_classes constraints to guarantee that prevalences_i>=0, and
            an additional constraints for ensuring that sum(prevalences)==1

        b : array, shape (n_constraints,)

        quadprog.solve_qp solves the following kind of problems:

        Minimize     1/2 x^T G x  a^T x
        Subject to   C.T x >= b

        Thus, the values of G, C and b must be the following

        G = train_distrib.T train_distrib
        C = [[ 1, 1, ...,  1],
             [ 1, 0, ...,  0],
             [ 0, 1, 0,.., 0],
             ...
             [ 0, 0, ..,0, 1]].T
        C shape (n_classes+1, n_classes)
        b = [1, 0, ..., 0]
        b shape (n_classes, )
    """
    G = 2 * train_distrib.T.dot(train_distrib)
    if not is_pd(G):
        G = nearest_pd(G)
    #  constraints, sum prevalences = 1, every prevalence >=0
    n_classes = len(classes)
    C = np.vstack([np.ones((1, n_classes)), np.eye(n_classes)]).T
    b = np.array([1] + [0] * n_classes, dtype=float)
    return G, C, b


############
# Functions to check if a matrix is positive definite and to compute the nearest positive definite matrix
# if it is not
############
def nearest_pd(A):
    """ Find the nearest positive-definite matrix to input

        A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which credits [2].

        References
        ----------
        [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

        [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
        matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if is_pd(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    indendity_matrix = np.eye(A.shape[0])
    k = 1
    while not is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += indendity_matrix * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def dpofa(m):
    """ Factors a symmetric positive definite matrix

        This is a version of the dpofa function included in quadprog library. Here, it is mainly used to check
        whether a matrix is positive definite or not

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to be factored. Only the diagonal and upper triangle are used

        Returns
        -------
        k : int,
            == 0  m is positive definite and the factorization has been completed
            >  0  the leading minor of order k is not positive definite

        r : array, an upper triangular matrix
            When k==0, the factorization is complete and r.T.dot(r) == m
            The strict lower triangle is unaltered (it is equal to the strict lower triangle of matrix m), so it
            could be different from 0.
   """
    r = np.array(m, copy=True)
    n = len(r)
    for k in range(n):
        s = 0.0
        if k >= 1:
            for i in range(k):
                t = r[i, k]
                if i > 0:
                    t = t - np.sum(r[0:i, i] * r[0:i, k])
                t = t / r[i, i]
                r[i, k] = t
                s = s + t * t
        s = r[k, k] - s
        if s <= 0.0:
            return k+1, r
        r[k, k] = np.sqrt(s)
    return 0, r


def is_pd(m):
    """ Checks whether a matrix is positive definite or not

        It is based on dpofa function, a version of the dpofa function included in quadprog library. When dpofa
        returns 0 the matrix is positive definite.

        Parameters
        ----------
        m : symmetric matrix, typically the shape is (n_classes, n_classes)
            The matrix to check whether it is positive definite or not

        Returns
        -------
        A boolean, True when m is positive definite and False otherwise

    """
    return dpofa(m)[0] == 0


############
# Functions for solving ED-based methods based on QUADPROG
############
def solve_ed(G, a, C, b):
    """ Solves the optimization problem for ED-based quantifiers

        It just calls `quadprog.solve_qp` with the appropriate parameters. These paremeters were computed
        before by calling `compute_ed_param_train` and `compute_ed_param_test`.
        In the derivation of the optimization problem, the last class is put in terms of the rest of classes. Thus,
        we have to add 1-prevalences.sum() which it is the prevalence of the last class

        Parameters
        ----------
        G : array, shape (n_classes, n_classes)

        C : array, shape (n_classes, n_constraints)
            n_constraints will be n_classes + 1

        b : array, shape (n_constraints,)

        a : array, shape (n_classes, )

        G, C and b are computed by `compute_ed_param_train` and a by `compute_ed_param_test`

        Returns
        -------
        prevalences : array, shape=(n_classes, )
           Vector containing the predicted prevalence for each class

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    sol = quadprog.solve_qp(G=G, a=a, C=C, b=b)
    prevalences = sol[0]
    # the last class was removed from the problem, its prevalence is 1 - the sum of prevalences for the other classes
    return np.append(prevalences, 1 - prevalences.sum())


def compute_ed_param_train(distance_func, train_distrib, classes, n_cls_i):
    """ Computes params related to the train distribution for solving ED-problems using `quadprog.solve_qp`

        Parameters
        ----------
        distance_func : function
            The function used to measure the distance between each pair of examples

        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        n_cls_i: ndarray, shape (n_classes, )
            The number of examples of each class

        Returns
        -------
        K : array, shape (n_classes, n_classes)
            Average distance between each pair of classes in the training set

        G : array, shape (n_classes - 1, n_classes - 1)

        C : array, shape (n_classes - 1, n_constraints)
            n_constraints will be equal to the number of classes (n_classes)

        b : array, shape (n_constraints,)

        See references below for further details

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    n_classes = len(classes)
    #  computing sum de distances for each pair of classes
    K = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        K[i, i] = distance_func(train_distrib[classes[i]], train_distrib[classes[i]]).sum()
        for j in range(i + 1, n_classes):
            K[i, j] = distance_func(train_distrib[classes[i]], train_distrib[classes[j]]).sum()
            K[j, i] = K[i, j]

    #  average distance
    K = K / np.dot(n_cls_i, n_cls_i.T)

    B = np.zeros((n_classes - 1, n_classes - 1))
    for i in range(n_classes - 1):
        B[i, i] = - K[i, i] - K[-1, -1] + 2 * K[i, -1]
        for j in range(n_classes - 1):
            if j == i:
                continue
            B[i, j] = - K[i, j] - K[-1, -1] + K[i, -1] + K[j, -1]

    #  computing the terms for the optimization problem
    G = 2 * B
    if not is_pd(G):
        G = nearest_pd(G)

    C = -np.vstack([np.ones((1, n_classes - 1)), -np.eye(n_classes - 1)]).T
    b = -np.array([1] + [0] * (n_classes - 1), dtype=float)

    return K, G, C, b


def compute_ed_param_test(distance_func, train_distrib, test_distrib, K, classes, n_cls_i):
    """ Computes params related to the test distribution for solving ED-problems using `quadprog.solve_qp`

        Parameters
        ----------
        distance_func : function
            The function used to measure the distance between each pair of examples

        train_distrib : array, shape (n_bins * n_classes, n_classes)
            Represents the distribution of each class in the training set

        test_distrib : array, shape (n_bins * n_classes, 1)
            Represents the distribution of the testing set

        K : array, shape (n_classes, n_classes)
            Average distance between each pair of classes in the training set

        classes : ndarray, shape (n_classes, )
            Class labels

        n_cls_i: ndarray, shape (n_classes, )
            The number of examples of each class

        Returns
        -------
        a : array, shape (n_classes, )
            Term a for solving optimization problems using `quadprog.solve_qp`

        See references below for further details

        References
        ----------
        Alberto Castaño, Laura Morán-Fernández, Jaime Alonso, Verónica Bolón-Canedo, Amparo Alonso-Betanzos,
        Juan José del Coz: An analysis of quantification methods based on matching distributions

        Hideko Kawakubo, Marthinus Christoffel Du Plessis, and Masashi Sugiyama. 2016. Computationally efficient
        class-prior estimation under class balance change using energy distance. Transactions on Information
        and Systems 99, 1 (2016), 176–186.
    """
    n_classes = len(classes)
    Kt = np.zeros(n_classes)
    for i in range(n_classes):
        Kt[i] = distance_func(train_distrib[classes[i]], test_distrib).sum()

    Kt = Kt / (n_cls_i.squeeze() * float(len(test_distrib)))

    a = 2 * (- Kt[:-1] + K[:-1, -1] + Kt[-1] - K[-1, -1])
    return a
