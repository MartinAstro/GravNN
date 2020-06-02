#include "PinesAlgorithm.h"
#include <iostream>
#include <cmath>



PinesAlgorithm::PinesAlgorithm(double r0, double muBdy, int degree)
{
    /*
    Want to regression object to take inputs: positions (cartesian) and accelerations (cartesian?)
    Want to regression object to output coefficient list
    */
    N = degree;
    a = r0;
    mu = muBdy;

    std::vector<double> doubleFiller;
    doubleFiller.resize(N+2, 0);

    rE.resize(N+2, 0);
    iM.resize(N+2,0);
    rhol.resize(N+2, 0);

    for(unsigned int i = 0; i <= degree + 1; i++)
    {
        std::vector<double> aRow, n1Row, n2Row;
        aRow.resize(i+1, 0.0);
        // Diagonal elements of A_bar
        if (i == 0)
        {
             aRow[i] = 1.0;
        }
        else
        {
            aRow[i] = sqrt(double((2*i+1)*getK(i))/(2*i*getK(i-1))) * aBar[i-1][i-1];
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
        aBar.push_back(aRow);
    }


    for (unsigned int l = 0; l <= N; l++) // up to _maxDegree-1
    {
        std::vector<double> nq1Row, nq2Row;
        nq1Row.resize(l+1, 0.0);
        nq2Row.resize(l+1, 0.0);
        for (unsigned int m = 0; m <= l; m++)
        {
            if (m < l)
            {
                nq1Row[m] = sqrt(double((l-m)*getK(m)*(l+m+1))/getK(m+1));
            }
            nq2Row[m] = sqrt(double((l+m+2)*(l+m+1)*(2*l+1)*getK(m))/((2*l+3)*getK(m+1)));
        }
        n1q.push_back(nq1Row);
        n2q.push_back(nq2Row);
    }

}

std::vector<double> PinesAlgorithm::compute_acc(std::vector<double> positions, std::vector<std::vector<double> > cBar, std::vector<std::vector<double> > sBar)
{
    double order;
    double rho;
    double a1, a2, a3, a4, sum_a1, sum_a2, sum_a3, sum_a4;
    std::vector<double> acc;
    order = N;

    double x, y, z;
    double u, t, s, r;
    acc.resize(positions.size(), 0);

    for (int p = 0; p < positions.size()/3; p++)
    {
        x = positions[3*p + 0];
        y = positions[3*p + 1];
        z = positions[3*p + 2];

        r = sqrt(pow(x,2) + pow(y,2) + pow(z, 2));
        s = x/r;
        t = y/r;
        u = z/r;

        for (unsigned int l = 1; l <= N+1; l++)
        {
            aBar[l][l-1] = sqrt(double((2*l)*getK(l-1))/getK(l)) * aBar[l][l] * u;
        }

        // Lower terms of A_bar
        for (unsigned int m = 0; m <= order+1; m++)
        {
            for(unsigned int l = m + 2; l <= N+1; l++)
            {
                aBar[l][m] = u * n1[l][m] * aBar[l-1][m] - n2[l][m] * aBar[l-2][m];
            }

            // Computation of real and imaginary parts of (2+j*t)^m
            if (m == 0)
            {
                rE[m] = 1.0;
                iM[m] = 0.0;
            } else {
                rE[m] = s * rE[m-1] - t * iM[m-1];
                iM[m] = s * iM[m-1] + t * rE[m-1];
            }
        }

        rho = a/r;
        rhol[0] = mu/r;
        rhol[1] = rhol[0]*rho;

        a1 = 0;
        a2 = 0; 
        a3 = 0;
        a4 = 0;
        for (unsigned int l = 1; l <= N; l++) // does not include l = maxDegree
        {
            rhol[l+1] =  rho * rhol[l]; // rho_l computed

            sum_a1 = 0;
            sum_a2 = 0;
            sum_a3 = 0;
            sum_a4 = 0;
            for(unsigned int m = 0; m <= l; m++)
            {
                double D, E, F;
                D = cBar[l][m] * rE[m] + sBar[l][m] * iM[m];
                if (m == 0)
                {
                    E = 0.0;
                    F = 0.0;
                }
                else
                {
                    E = cBar[l][m] * rE[m-1] + sBar[l][m] * iM[m-1];
                    F = sBar[l][m] * rE[m-1] - cBar[l][m] * iM[m-1];
                }

                sum_a1 = sum_a1 + m * aBar[l][m] * E;
                sum_a2 = sum_a2 + m * aBar[l][m] * F;
                if (m < l)
                {
                    sum_a3 = sum_a3 + n1q[l][m] * aBar[l][m+1] * D; // The reason for this is from Eq. 79 in the documentation -- numerator goes to zero
                }
                sum_a4 = sum_a4 + n2q[l][m] * aBar[l+1][m+1] * D;
            }

            a1 += rhol[l+1]/a * sum_a1;
            a2 += rhol[l+1]/a * sum_a2;
            a3 += rhol[l+1]/a * sum_a3;
            a4 -= rhol[l+1]/a * sum_a4;
        }
        a4 -= rhol[1]/a;

        acc[3*p + 0] = a1 + s * a4;
        acc[3*p + 1] = a2 + t * a4;
        acc[3*p + 2] = a3 + u * a4;
    }
	
    return acc;
}
PinesAlgorithm::~PinesAlgorithm()
{
}
