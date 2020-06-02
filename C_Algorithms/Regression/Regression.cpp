#include "Regression.h"
#include <iostream>

Regression::Regression(std::vector<double> rVec1D, std::vector<double> aVec1D, int degree, double r0, double muBdy)
{
    /*
    Want to regression object to take inputs: positions (cartesian) and accelerations (cartesian?)
    Want to regression object to output coefficient list
    */
    P = rVec1D.size();
    N = degree;
    a = r0;
    mu = muBdy;

    std::vector<double> doubleFiller;
    doubleFiller.resize(N+2, 0);

    r.resize(N+2, 0);
    i.resize(N+2,0);
    rho.resize(N+2, 0);
    coeff = Eigen::VectorXd::Zero(N*(N+1));
    A.resize(N+2, doubleFiller);

    for(unsigned int i = 0; i < N + 2; i++)
    {
        std::vector<double> aRow, n1Row, n2Row;
        aRow.resize(i+1, 0.0);
        // Diagonal elements of A_bar
        if (i == 0)
        {
            A[i][i] = 1.0;
        }
        else
        {
            A[i][i] = sqrt(double((2*i+1)*getK(i))/(2*i*getK(i-1))) * A[i-1][i-1];
        }
        n1Row.resize(i+1, 0.0);
        n2Row.resize(i+1, 0.0);
        for (unsigned int m = 0; m <= i; m++)
        {
            if (i >= m + 2)
            {
                n1Row[m] = sqrt(double((2*i+1)*(2*i-1))/((i-m)*(i+m)));
                n2Row[m] = sqrt(double((i+m-1)*(2*i+1)*(i-m-1))/((i+m)*(i-m)*(2*i-3)));

            }
        }
        n1.push_back(n1Row);
        n2.push_back(n2Row);
    }
    

    positions = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(rVec1D.data(), rVec1D.size());
    accelerations =Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(aVec1D.data(), aVec1D.size());
}

void Regression::populate_variables(double x, double y, double z)
{
    double rMag = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    s = x/rMag;
    t = y/rMag;
    u = z/rMag;

    //Eq 23
    A[0][0] = 1;
    for (int n = 1; n < A.size(); n++)
    {
         A[n][n-1] = sqrt(double((2*n)*getK(n-1))/getK(n)) * A[n][n] * u;
    }

    for (int n = 2; n < A.size(); n++)
    {
        //Eq 21
        for (int m = 0; m < A.size(); m++)
        {
            A[n][m] = u * n1[n][m] * A[n-1][m] - n2[n][m] * A[n-2][m];
        }
    }

    //Eq 24
    r[0] = 1; // cos(m*lambda)*cos(m*alpha);
    i[0] = 0; //sin(m*lambda)*cos(m*alpha);
    for (int m = 1; m < r.size(); m++)
    {
        r[m] = s*r[m-1] - t*i[m-1];
        i[m] = s*i[m-1] + t*r[m-1];
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

void Regression::perform_regression()
{
    Eigen::MatrixXd M(P, N*(N+1));

    double f_Cnm_1, f_Cnm_2, f_Cnm_3;
    double f_Snm_1, f_Snm_2, f_Snm_3;
    double n_lm_n_lm_p1; // Eq 79 BSK
    double n_lm_n_l_p1_m_p1; // Eq 80 BSK
    int delta_m, delta_m_p1;
    double c1, c2;
    double rTerm, iTerm;
    for (int p = 0; p < P/3; p++)
    {  
        populate_variables(positions(3*p), positions(3*p + 1), positions(3*p + 2));
        for (int n = 0; n < N; n++)
        {
            for (int m = 0; m < n; m++)
            {
                delta_m = (m == 0) ? 1 : 0;
                delta_m_p1 = (m+1 == 0) ? 1: 0;
                n_lm_n_lm_p1 = sqrt((n-m)*(2-delta_m)*(n+m+1)/(2-delta_m_p1));
                n_lm_n_l_p1_m_p1 = sqrt((n+m+2)*(n+m+1)*(2*n+1)*(2-delta_m)/((2*n+3)*(2-delta_m_p1)));

                c1 = n_lm_n_lm_p1;
                c2 = n_lm_n_l_p1_m_p1;

                //TODO: These will need the normalizaiton factor out in front (N1, N2)
                // Coefficient contribution to X, Y, Z components of the acceleration
                
                if (m == 0) { 
                    rTerm = 0;
                    iTerm = 0;
                } else {
                    rTerm = r[m-1];
                    iTerm = i[m-1];
                }
                f_Cnm_1 = (rho[n+2]/a)*(m*A[n][m]*rTerm -s*c2*A[n+1][m+1]*r[m]);
                f_Cnm_2 = -(rho[n+2]/a)*(m*A[n][m]*rTerm + t*c2*A[n+1][m+1]*r[m]);
                f_Cnm_3 = (rho[n+2]/a)*(c1*A[n][m+1] - u*c2*A[n+1][m+1])*r[m];

                f_Snm_1 = (rho[n+2]/a)*(m*A[n][m]*iTerm -s*c2*A[n+1][m+1]*i[m]);
                f_Snm_2 = (rho[n+2]/a)*(m*A[n][m]*iTerm - t*c2*A[n+1][m+1]*i[m]);
                f_Snm_3 = (rho[n+2]/a)*(c1*A[n][m+1] - u*c2*A[n+1][m+1])*i[m];
                
                
                M(3*p + 0, n*(n+1) + 2*m + 0) = f_Cnm_1; // X direction
                M(3*p + 0, n*(n+1) + 2*m + 1) = f_Snm_1;
                M(3*p + 1, n*(n+1) + 2*m + 0) = f_Cnm_2; // Y direction
                M(3*p + 1, n*(n+1) + 2*m + 1) = f_Snm_2;
                M(3*p + 2, n*(n+1) + 2*m + 0) = f_Cnm_3; // Z direction
                M(3*p + 2, n*(n+1) + 2*m + 1) = f_Snm_3;
            }
        }
    }

    //std::cout << "The solution using the QR decomposition is:\n" << M.colPivHouseholderQr().solve(accelerations) << std::endl;
    std::cout << "The solution using the QR decomposition is:\n" << M.fullPivHouseholderQr().solve(accelerations) << std::endl;
    std::cout << "The least-squares solution is:\n"<< M.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(accelerations) << std::endl;

}

Regression::~Regression()
{
}
