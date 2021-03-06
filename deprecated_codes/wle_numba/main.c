/*
 *  main.c
 *
 *
 *  Created by Earl Lawrence on 9/17/09.
 *
 *  This program was prepared by Los Alamos National Security, LLC at Los Alamos National Laboratory (LANL)
 *  under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy (DOE). All rights in the program
 *  are reserved by the DOE and Los Alamos National Security, LLC.  Permission is granted to the public to
 *  copy and use this software without charge, provided that this Notice and any statement of authorship are
 *  reproduced on all copies.  Neither the U.S. Government nor LANS makes any warranty, express or implied,
 *  or assumes any liability or responsibility for the use of this software.
 *
 */


#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"

int main(int argc, char **argv) {
    int i,j,type=2, writeout=0;
    double xstar[7], ystar[2*582], stuff[4], xstarcmb[6];
    FILE *fp;

    char * fname_nl = argv[1];
    int cmbh=0;
	xstar[0] = atof(argv[2]); // Omega_b*h^2
    xstar[1] = atof(argv[3]); // Omega_m*h^2
    xstar[2] = atof(argv[4]); // n_s
	xstar[3] = atof(argv[5]); // H0
	xstar[4] = atof(argv[6]); // w
	xstar[5] = atof(argv[7]); // sigma_8
	xstar[6] = atof(argv[8]); // z
	// 0: Delta^2/k^1.5; 1: Delta^2; 2: P(k))

    if(cmbh == 1) {
        xstarcmb[0] = xstar[0];
        xstarcmb[1] = xstar[1];
        xstarcmb[2] = xstar[2];
        xstarcmb[3] = xstar[4];
        xstarcmb[4] = xstar[5];
        xstarcmb[5] = xstar[6];
        emu_noh(xstarcmb, ystar, &type);
        getH0fromCMB(xstarcmb, stuff);
        xstar[3] = 100.*stuff[3];
    } else {
        emu(xstar, ystar, &type);
    }

    // Write the nonlinear file
    if ((fp = fopen(fname_nl,"w"))==NULL) {
        printf("cannot open %s \n",fname_nl);
        exit(1);
    }

    fprintf(fp, "# Parameters:\n");
    fprintf(fp, "# omega_b = %f, omega_m = %f, n_s = %f, h = %f, w = %f, sigma_8 = %f\n", xstar[0], xstar[1], xstar[2], xstar[3], xstar[4], xstar[5]);
    fprintf(fp, "# z = %f\n", xstar[6]);
    fprintf(fp, "#\n");
    fprintf(fp, "# k[1/Mpc] ");

    switch(type) {
        default:
            fprintf(fp, "# Delta^2 / k^1.5:\n");
            break;
        case 1:
            fprintf(fp, "# Delta^2:\n");
            break;
        case 2:
            fprintf(fp, "# P(k):\n");
            break;
    }

    for(j=0; j<582; j++) {
        fprintf(fp ,"%e %e \n", ystar[j], ystar[582+j]);
    }
    fclose(fp);
}
