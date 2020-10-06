#include "PinesAlgorithm.h"
#include <iostream>
#include <cmath>
#include <thread>
#include <future>
#include <omp.h>


PinesAlgorithm::PinesAlgorithm(double r0, double muBdy, int degree, std::vector<std::vector<double> > cBarIn, std::vector<std::vector<double> > sBarIn)
{
    /*
    Want to regression object to take inputs: positions (cartesian) and accelerations (cartesian?)
    Want to regression object to output coefficient list
    */
    N = degree;
    a = r0;
    mu = muBdy;
    this->cBar = cBarIn;
    this->sBar = sBarIn;

    std::vector<double> doubleFiller;
    doubleFiller.resize(N+2, 0);


    for(unsigned int i = 0; i <= degree + 1; i++)
    {
        std::vector<double> aRow, n1Row, n2Row;
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

int PinesAlgorithm::print_percentage(int n, int size, int progress)
{
    int percent = (int)round(float(n) / float(size)*100.0);
    if (percent > progress)
    {
        std::cout << progress << std::endl;
        progress += 10;
    }
    return progress;
}

std::vector<double> PinesAlgorithm::compute_acc(std::vector<double> positions)
{
    std::vector<double> acc(positions.size(), 0);
    int progress = 0;
    double order = N;
    std::vector<double> acc_inst(3, 0);
    int num_threads =  std::thread::hardware_concurrency();
    
    omp_set_num_threads(num_threads); 
    int p;
    #pragma omp parallel for private(p)
    for (p = 0; p < positions.size()/3; p++)
    {
        progress = print_percentage(p, positions.size()/3, progress);
        acc_inst = compute_acc_thread(positions[3*p + 0], positions[3*p + 1], positions[3*p + 2]);
        //std::future<std::vector<double> > result = std::async(std::launch::async, &PinesAlgorithm::compute_acc_thread, this, positions[3*p + 0], positions[3*p + 1], positions[3*p + 2]);
        //acc_inst = result.get();
        acc[3*p+0] = acc_inst[0]; 
        acc[3*p+1] = acc_inst[1];
        acc[3*p+2] = acc_inst[2];
    }
    return acc;
}


std::vector<double> PinesAlgorithm::compute_acc_thread(double x, double y, double z)
{
    double rho;
    double a1, a2, a3, a4;
    double sum_a1, sum_a2, sum_a3, sum_a4;
    double u, t, s, r;
    double D, E, F;
    std::vector<double> rhol(N+2,0);
    std::vector<double> rE(N+2, 0), iM(N+2,0);
    std::vector<std::vector<double> > aBar;
    std::vector<double> dubFiller;
    std::vector<double> acc(3,0);
    dubFiller.resize(N+2, 0);
    aBar.resize(N+2, dubFiller);
   
    if (cBar[0][0] == 0.0)
    {
        return acc;
    }

    r = sqrt(pow(x,2) + pow(y,2) + pow(z, 2));
    s = x/r;
    t = y/r;
    u = z/r;

    aBar[0][0] = 1.0;
    for (unsigned int l = 1; l <= N+1; l++)
    {
        aBar[l][l] = sqrt(double((2*l+1)*getK(l))/(2*l*getK(l-1))) * aBar[l-1][l-1];
        aBar[l][l-1] = sqrt(double((2*l)*getK(l-1))/getK(l)) * aBar[l][l] * u;
    }

    // Lower terms of A_bar
    for (unsigned int m = 0; m <= N+1; m++)
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

    a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    for (unsigned int l = 1; l <= N; l++) // does not include l = maxDegree
    {
        rhol[l+1] =  rho * rhol[l]; // rho_l computed

        sum_a1 = 0, sum_a2 = 0, sum_a3 = 0, sum_a4 = 0;
        for(unsigned int m = 0; m <= l; m++)
        {
            D = cBar[l][m] * rE[m] + sBar[l][m] * iM[m];
            E = (m == 0) ? 0.0 : cBar[l][m] * rE[m-1] + sBar[l][m] * iM[m-1];
            F = (m == 0) ? 0.0 : sBar[l][m] * rE[m-1] - cBar[l][m] * iM[m-1];

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

    acc[0] = a1 + s * a4;
    acc[1] = a2 + t * a4;
    acc[2] = a3 + u * a4;

    if (std::isnan(acc[0]))
    {
        std::cout << "WE GOT EM\n";
    }
    return acc;
}

std::vector<double> PinesAlgorithm::compute_acc_components(std::vector<double> positions)
{
    int progress;
    double order;
    double rho;
    double a1, a2, a3, a4;
    double sum_a1, sum_a2, sum_a3, sum_a4;
    double x, y, z;
    double u, t, s, r;
    double D, E, F;
    int idx;
    int total_components = (N)*(N+1)/2*3;
    std::vector<double> rhol;
    std::vector<double> rE, iM;
    std::vector<std::vector<double> > aBar;
    std::vector<double> acc_components(((positions.size()/3))*total_components, 0);
    std::cout << total_components << "\n";

    order = N;
    progress = 0;

    if (cBar[0][0] == 0.0)
    {
        return acc_components;
    }

    for (int p = 0; p < positions.size()/3; p++)
    {
        progress = print_percentage(p, positions.size()/3, progress);
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

        a1 = 0, a2 = 0, a3 = 0, a4 = 0;
        for (unsigned int l = 1; l < N; l++) // does not include l = maxDegree
        {
            rhol[l+1] =  rho * rhol[l]; // rho_l computed

            for(unsigned int m = 0; m <= l; m++)
            {
                D = cBar[l][m] * rE[m] + sBar[l][m] * iM[m];
                E = (m == 0) ? 0.0 : cBar[l][m] * rE[m-1] + sBar[l][m] * iM[m-1];
                F = (m == 0) ? 0.0 : sBar[l][m] * rE[m-1] - cBar[l][m] * iM[m-1];

                idx = total_components*p + 3*(l*(l+1)/2) + 3*m;
                // std::cout << total_components << "\t" << l*(l+1)/2 << "\t" << 3*(l*(l+1)/2) << "\t" << idx <<  "\n";
                // std::cin.get(); 

                a4 = -1.0* rhol[l+1]/a *n2q[l][m] * aBar[l+1][m+1] * D;

                acc_components[idx+0] = rhol[l+1]/a* m*aBar[l][m]*E + s*a4;
                acc_components[idx+1] = rhol[l+1]/a* m*aBar[l][m]*F + t*a4;

                if (m < l)
                {
                    acc_components[idx+2] = rhol[l+1]/a* n1q[l][m] * aBar[l][m+1] * D + u*a4;
                }
            }
        }
    }
    return acc_components;
}

PinesAlgorithm::~PinesAlgorithm()
{
}
