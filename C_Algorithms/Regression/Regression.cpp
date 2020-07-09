#include "Regression.h"
#include <iostream>
#include <cmath>

#include "../PinesAlgorithm/PinesAlgorithm.h"


Regression::Regression(std::vector<double> rVec1D, std::vector<double> aVec1D, int degree, double r0, double muBdy)
{
    // initialize variables
    this->P = rVec1D.size();
    this->N = degree;
    this->a = r0;
    this->mu = muBdy;

    this->pos_meas_eigen = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(rVec1D.data(), rVec1D.size());
    this->acc_meas_eigen =Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aVec1D.data(), aVec1D.size());
    this->pos_meas = rVec1D;
    this->acc_meas = aVec1D;

    std::vector<double> coef_row;
    for (int l = 0; l <= this->N; l++)
    {
        coef_row.resize(l+1, 0.0);
        coef_regress.C_lm.push_back(coef_row);
        coef_regress.S_lm.push_back(coef_row);
    }

    // Load single calculation objects
    std::vector<double> dubFiller(N+2, 0);
    std::vector<std::vector<double> > empty(N+2, dubFiller);
    this->A = empty;
    this->r = dubFiller;
    this->i = dubFiller;
    this->rho.resize(N+2+1, 0); // Need to go to + 3 for the original rho[n+2] formulation. 

    for(unsigned int l = 0; l < N+2; l++)
    {
        this->A[l][l] = (l == 0) ? 1.0 : sqrt((2*l+1)*getK(l))/(2*l*getK(l-1)) * this->A[l-1][l-1]; // Diagonal elements of A_bar
        
        std::vector<double> n1Row(l+1, 0.0);
        std::vector<double> n2Row(l+1, 0.0);
        for (unsigned int m = 0; m <= l; m++)
        {
            if (l >= m + 2)
            {
                n1Row[m] = sqrt((2.0*l+1.0)*(2.0*l-1))/((l-m)*(l+m));
                n2Row[m] = sqrt((l+m-1.0)*(2.0*l+1.0)*(l-m-1.0))/((l+m)*(l-m)*(2.0*l-3.0));
            }
        }
        this->n1.push_back(n1Row);
        this->n2.push_back(n2Row);
    }
}

void Regression::set_regression_variables(double x, double y, double z)
{
    double rMag = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    this->s = x/rMag;
    this->t = y/rMag;
    this->u = z/rMag;

    //Eq 23
    for (unsigned int l = 1; l < A.size(); l++)
    {
         this->A[l][l-1] = sqrt((2.0*l)*getK(l-1)/getK(l)) * this->A[l][l] * this->u;
    }

    for (unsigned int m = 0; m < this->A.size(); m++)
    {
        for(unsigned int l = m + 2; l < this->A.size(); l++)
        {
            this->A[l][m] = u *this-> n1[l][m] * this->A[l-1][m] - this->n2[l][m] * this->A[l-2][m];
        }
    }

    //Eq 24
    this->r[0] = 1; // cos(m*lambda)*cos(m*alpha);
    this->i[0] = 0; //sin(m*lambda)*cos(m*alpha);
    for (int m = 1; m < r.size(); m++)
    {
        this->r[m] = this->s*this->r[m-1] - this->t*this->i[m-1];
        this->i[m] = this->s*this->i[m-1] + this->t*this->r[m-1];
    }

    //Eq 26 and 26a
    double beta = a/rMag;
    rho[0] = mu/rMag;
    rho[1] = rho[0]*beta;
    for(int n = 2; n < rho.size(); n++)
    {
        rho[n] = beta*rho[n-1];
    }
}

Coefficients Regression::perform_regression(int l_i, double precision)
{   
    this->l_i = l_i; // Degree of first coefficient to regress (typically either 0 or 2)
    this->M_skip = l_i*(l_i+1); // Number of coefficients to skip
    this->M_total = (this->N + 1)*(this->N + 2); // Total number of coefficients up to degree l_{f+1} -- e.g if l_max = 2 then M_total = 12

    Eigen::MatrixXd M(this->P, M_total - M_skip);
    Eigen::SparseMatrix<double> M_sparse(this->P, M_total - M_skip);

    int delta_m, delta_m_p1;
    double f_Cnm_1, f_Cnm_2, f_Cnm_3;
    double f_Snm_1, f_Snm_2, f_Snm_3;
    double c1, c2; // Eq 79, 80 BSK
    double rTerm, iTerm;
    double x, y, z;
    int idx;
    int degIdx;
    for (int p = 0; p < P/3; p++)
    {  
        x = pos_meas_eigen(3*p);
        y = pos_meas_eigen(3*p + 1);
        z = pos_meas_eigen(3*p + 2);
        this->set_regression_variables(x, y, z);
        for (int l = l_i; l <= N; l++)
        {
            for (int m = 0; m <= l; m++)
            {
                delta_m = (m == 0) ? 1 : 0;
                delta_m_p1 = (m+1 == 0) ? 1: 0;
                c1 = sqrt((l-m)*(2.0-delta_m)*(l+m+1.0)/(2.0-delta_m_p1)); // n_lm_n_lm_p1
                c2 = sqrt((l+m+2.0)*(l+m+1)*(2.0*l+1.0)*(2.0-delta_m)/((2.0*l+3.0)*(2.0-delta_m_p1))); // n_lm_n_l_p1_m_p1

                rTerm =(m == 0) ? 0 : r[m-1];
                iTerm =(m == 0) ? 0 : i[m-1];

                // Coefficient contribution to X, Y, Z components of the acceleration
                // ORIGINAL CALL FROM PAPER
                f_Cnm_1 = (rho[l+2]/a)*(m*A[l][m]*rTerm - s*c2*A[l+1][m+1]*r[m]);
                f_Cnm_2 = -(rho[l+2]/a)*(m*A[l][m]*rTerm + t*c2*A[l+1][m+1]*r[m]);
                f_Cnm_3 = (rho[l+2]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*r[m];

                f_Snm_1 = (rho[l+2]/a)*(m*A[l][m]*iTerm - s*c2*A[l+1][m+1]*i[m]);
                f_Snm_2 = (rho[l+2]/a)*(m*A[l][m]*iTerm - t*c2*A[l+1][m+1]*i[m]);
                f_Snm_3 = (rho[l+2]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*i[m];

                // RHO N+1 RATHER THAN N+2 BC CAN'T FIGURE OUT MATH
                // f_Cnm_1 = (rho[l+1]/a)*(m*A[l][m]*rTerm - s*c2*A[l+1][m+1]*r[m]);
                // f_Cnm_2 = -(rho[l+1]/a)*(m*A[l][m]*rTerm + t*c2*A[l+1][m+1]*r[m]);
                // f_Cnm_3 = (rho[l+1]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*r[m];

                // f_Snm_1 = (rho[l+1]/a)*(m*A[l][m]*iTerm - s*c2*A[l+1][m+1]*i[m]);
                // f_Snm_2 = (rho[l+1]/a)*(m*A[l][m]*iTerm - t*c2*A[l+1][m+1]*i[m]);
                // f_Snm_3 = (rho[l+1]/a)*(c1*A[l][m+1] - u*c2*A[l+1][m+1])*i[m];

                degIdx = l*(l+1) - M_skip;
                M(3*p + 0, degIdx + 2*m + 0) = f_Cnm_1; // X direction
                M(3*p + 0, degIdx + 2*m + 1) = f_Snm_1;
                M(3*p + 1, degIdx + 2*m + 0) = f_Cnm_2; // Y direction
                M(3*p + 1, degIdx + 2*m + 1) = f_Snm_2;
                M(3*p + 2, degIdx + 2*m + 0) = f_Cnm_3; // Z direction
                M(3*p + 2, degIdx + 2*m + 1) = f_Snm_3;

                // M_sparse.insert(3*p + 0, degIdx + 2*m + 0) = f_Cnm_1; // X direction
                // M_sparse.insert(3*p + 0, degIdx + 2*m + 1) = f_Snm_1;
                // M_sparse.insert(3*p + 1, degIdx + 2*m + 0) = f_Cnm_2; // Y direction
                // M_sparse.insert(3*p + 1, degIdx + 2*m + 1) = f_Snm_2;
                // M_sparse.insert(3*p + 2, degIdx + 2*m + 0) = f_Cnm_3; // Z direction
                // M_sparse.insert(3*p + 2, degIdx + 2*m + 1) = f_Snm_3;
            }
        }
    }
    find_solution(precision, M, M_sparse);
    return coef_regress;
}

void Regression::find_solution(double precision, Eigen::MatrixXd &M, Eigen::SparseMatrix<double> &M_sparse)
{
    // Initialize variables
    int iterations = 0;
    int max_iterations = 1000;
    double rel_error;
    PinesAlgorithm pines = PinesAlgorithm(a, mu, N);
    std::vector<double> acc_regress;
    std::vector<double> coef_row;
    std::vector<double> zero_vec(acc_meas.size(), 0);

    Eigen::VectorXd Y = acc_meas_eigen; // true acceleration
    Eigen::VectorXd Y_hat; // regressed acceleration
    Eigen::VectorXd dY; // difference in acceleration
    Eigen::VectorXd X_f; // new coefficient guess 
    Eigen::VectorXd X_i= Eigen::VectorXd::Zero(M_total - M_skip); // original coefficient guess
    Eigen::VectorXd dX; // improvement for coefficients

    // Initialize variables and sparse matrix
    solution_unique = Y.size() > (M_total - M_skip); // Check if the problem is underdetermined
    solution_exists = false;

    // Perform one time calcuations
    // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solver;
    // M_sparse.makeCompressed();
    // solver.compute(M_sparse);

    while(!solution_exists && iterations < max_iterations)
    {
        acc_regress = pines.compute_acc(pos_meas, coef_regress.C_lm, coef_regress.S_lm);
        Y_hat = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(acc_regress.data(), acc_regress.size());
        dY =  Y - Y_hat;

        // try BDCSVD decomposition
        dX = M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(dY);
        //dX = M.fullPivHouseholderQr().solve(dY);
        //dX = solver.solve(dY);
        rel_error = (M*dX - dY).norm() / dY.norm();
        solution_exists = rel_error < precision;

        X_f = X_i + dX;
        // X_f(0) = 1.0;
        // X_f.segment(1,5) *= 0.0;
        format_coefficients(X_f);
        X_i = X_f;
        iterations++;
    }
    std::cout << "Ran " + std::to_string(iterations) + " iterations! \n";
    X_f(0) = 1.0;
}

void Regression::format_coefficients(Eigen::VectorXd coef_list)
{
    int l = this->l_i;
    int m = 0;
    this->coef_regress.C_lm[0][0] = (this->l_i != 0) ? 1.0 : 0.0; // If regressing only C20 and force the earlier coefficients

    for(int i = 0; i < coef_list.size()/2; i++)
    {
        if (m > l)
        {
            l += 1;
            m = 0;
        } 
        this->coef_regress.C_lm[l][m] = coef_list[2*i];
        this->coef_regress.S_lm[l][m] = coef_list[2*i + 1];
        m += 1;
    }
}

Regression::~Regression()
{
}
