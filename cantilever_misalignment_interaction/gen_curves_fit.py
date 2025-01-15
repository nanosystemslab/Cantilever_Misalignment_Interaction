#! /usr/bin/env python3

"""
Plots elliptic integral function over a range of wall angles.
2023-09-27
"""

import argparse
import glob
import logging
import os
import pathlib
import re
import sys
import types

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib import transforms
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import special
from scipy import constants


__version__ = "0.0.1"

#def elliptic_integral_function(deg,C4q,s):
def elliptic_integral_function(deg):

    #define wall angle
    theta=deg                           #degrees
    theta0=theta*np.pi/180              #radians

    #set up domains for allowed end (point B) angles
    w=100                               #number of end angle variations
    beta=theta0
    thetaBp = np.linspace(beta,np.pi/2.01+2*beta,w)

    thetaB=thetaBp-beta
    thetaB2=thetaB-beta

    #calculate (xB1/L,yB1/L) in (x1,y1)
    n=np.tan(thetaB)
    eta=1/np.cos(thetaB)
    k2=0.5
    gamma2=(np.pi/2)
    gamma1=np.arcsin((1/eta)*np.sqrt(eta/(eta+n)))

    e=special.ellipeinc(gamma2,k2)-special.ellipeinc(gamma1,k2)
    f=special.ellipkinc(gamma2,k2)-special.ellipkinc(gamma1,k2)

    alpha=f*(1+n**2)**(-1/4)
    aL=(1/(f*eta**2))*(-n*eta*f+2*n*eta*e+eta*np.sqrt(2*n/eta))
    bL=(1/(f*eta**2))*(eta*f-2*eta*e+eta*n*np.sqrt(2*n/eta))
    aL[0]=1
    bL[0]=0

    #transform to the (x2,y2) coordinate system
    ap=aL*np.cos(beta)+bL*np.sin(beta)
    bp=-aL*np.sin(beta)+bL*np.cos(beta)

    alpha2=np.sign(eta)*f/np.sqrt(abs(eta))

    #displacement from original position
    deltaB=np.tan(theta0)+bp/ap

    #C4 is the same as L*, dimensionless arc length
    C4=1/ap
    C3=np.cos(beta)+np.sin(beta)*np.tan(thetaB)

    #force parameter in (x2,y2)
    C1=abs(C3*(alpha**2)*(ap**2)/2)

    #initial value of C1 is 0, unloaded
    C1[0]=0

    return C1, C4, deltaB

def build_df(theta_range = None, theta_step=None):

    n = np.arange(-1*theta_range,theta_range+theta_step, theta_step)

    colorvary=np.linspace(0,1,len(n));
    c=np.empty([len(colorvary), 3])
    c[:,1] = 1 - colorvary
    c[:,2] = colorvary

    dataframes = []

    for i in range(len(n)):
        df = pd.DataFrame()
        C1, C4, deltaB = elliptic_integral_function(n[i])

        df["C1"]=C1
        df["C4"]=C4
        df["deltaB"]=deltaB
        df.meta = types.SimpleNamespace()
        df.meta.theta = n[i]

        dataframes.append(df)
    return dataframes

def poly_ellip_fits(deg=None, theta_range=None, theta_step=None):

    dataframes = build_df(theta_range, theta_step)

    index = np.arange(-theta_range,theta_range, theta_step)
    df_fits = pd.DataFrame(columns=["C1_fit", "C4_fit", "deltaB_lims"], index=index)

    for df in dataframes:
        C1_fit = np.polyfit(df["deltaB"], df["C1"], deg)#, full=True)
        C4_fit = np.polyfit(df["deltaB"], df["C4"], deg)#, full=True)
        df_fits["C1_fit"][df.meta.theta] = C1_fit
        df_fits["C4_fit"][df.meta.theta] = C4_fit
        df_fits["deltaB_lims"][df.meta.theta] = [df["deltaB"].min(), df["deltaB"].max()]

    out_path = "data/Fit_Curve_theta{}-{}-step-{}.h5".format(
            -1*theta_range, theta_range, theta_step)
    df_fits.to_hdf(out_path, key='df', mode='w')

    print("Data has been expanded and saved to Excel.")
    out_path_excel = "data/Fit_Curve_theta{}-{}-step-{}.xlsx".format(-1*theta_range, theta_range, theta_step)
    df_fits.to_excel(out_path_excel, index=True)  # Set index=False if you do not want to write row numbers


    return df_fits


def plot_data_output_C1_vs_deltaB(param_y="T", param_x="T", dataframes=None):
    """
    plot one data trace from a data frame

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "C1":{'label':"Normalized Downward Force", 'hi':300, 'lo':0, 'lbl':"Normalized Downward Force"},
        "deltaB":{'label':"Normalized Tip Deflection \delta_{\it B}^*", 'hi':300, 'lo':0, 'lbl':"Normalized Tip Deflection $\delta_{\it B}^*$"},
        "C4":{'label':"Normalized Cantilever Length {\it C}_4", 'hi':300, 'lo':0, 'lbl':"Normalized Cantilever Length ${\it C}_4$"},
    }


    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('plasma')

    num_lines = len(dataframes)

    with sns.axes_style("darkgrid"):
        #for df in dataframes:
        for i, df in enumerate(dataframes):
            x = df[param_x]
            y = df[param_y]
            #line = "{}-vs-{}".format(param_y, param_x)
            color = cmap(i / num_lines)
            ax.plot(x, y, color=color)

        # Adding curved arrow using FancyArrowPatch
        arrow = FancyArrowPatch((0.75, 0), (1.75, 0.8), connectionstyle="arc3,rad=-0.3",
                                arrowstyle='-|>', mutation_scale=20, color='black', lw=2.5, zorder=10)
        plt.gca().add_patch(arrow)
        plt.annotate('Increasing $\\theta_0$', xy=(1.5, 0.7), xytext=(1.20, 0.7),
                     fontsize=10, rotation=25, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # Adding the rotated text label (θ₀ = +60°)
        plt.text(1.7, 0.5, r'$\theta_0 = +60^\circ$', fontsize=14, rotation=290, fontweight='bold', zorder=12)

        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])
        ax.set_xlim(0,2.2)
        #ax.legend(frameon=True)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-output-{}-vs-{}.png".format(
            param_y, param_x)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

def plot_data_output(param_y="T", param_x="T", dataframes=None):
    """
    plot one data trace from a data frame

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "C1":{'label':"Normalized Downward Force", 'hi':300, 'lo':0, 'lbl':"Normalized Downward Force"},
        "deltaB":{'label':"Normalized Tip Deflection \delta_{\it B}^*", 'hi':300, 'lo':0, 'lbl':"Normalized Tip Deflection $\delta_{\it B}^*$"},
        "C4":{'label':"Normalized Cantilever Length {\it C}_4", 'hi':300, 'lo':0, 'lbl':"Normalized Cantilever Length ${\it L}^*$"},
    }


    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 4./3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('plasma')
    num_lines = len(dataframes)

    with sns.axes_style("darkgrid"):
        for i, df in enumerate(dataframes):
            x = df[param_x]
            y = df[param_y]
            color = cmap(i / num_lines)
            ax.plot(x, y, color=color)

        # Adding curved arrow using FancyArrowPatch
        arrow = FancyArrowPatch((0.5, 2.4), (1.9, 1.4), connectionstyle="arc3,rad=0.3",
                                arrowstyle='-|>', mutation_scale=20, color='black', lw=2.5, zorder=10)
        plt.gca().add_patch(arrow)
        plt.annotate('Increasing $\\theta_0$', xy=(1.5, 1.25), xytext=(1.1, 1.275),
                     fontsize=10, rotation=345, fontweight='bold',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

        # Adding the rotated text label (θ₀ = +60°)
        plt.text(1.85, 1.2, r'$\theta_0 = +60^\circ$', fontsize=14, rotation=50, fontweight='bold', zorder=12)
        plt.text(0.38, 2.45, r'$\theta_0 = -60^\circ$', fontsize=14, rotation=41, fontweight='bold', zorder=12)

        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])
        ax.set_xlim(0,2.5)
        ax.set_ylim(1,3)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-output-{}-vs-{}.png".format(
            param_y, param_x)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    # addl keys: asctime, module, name
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))

    return

def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    # -h/--help is auto-added
    parser.add_argument("-d", "--dir", dest="dirs",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        # action="store",
                        nargs='+',
                        default=None, required=False, help="path to input")
    parser.add_argument("-p", "--port", dest="port",
                        # action="store",
                        type= pathlib.Path,
                        nargs='+',
                        default=None, required=False, help="path to port")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])

    return ret

def format_poly(coeffs):
    terms = []
    degree = len(coeffs) - 1
    for i, coef in enumerate(coeffs):
        term_degree = degree - i
        if term_degree > 1:
            terms.append(f"{coef:.6f}x^{term_degree}")
        elif term_degree == 1:
            terms.append(f"{coef:.6f}x")
        else:
            terms.append(f"{coef:.6f}")
    return " + ".join(terms)

def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])
    logging.info(cmd_args)


    spread = 65
    step = 5
    dfs = build_df(theta_range=spread, theta_step=step)

    # plot_data_output(param_y="C1", param_x="deltaB", dataframes=dfs)
    # plot_data_output(param_y="C4", param_x="deltaB", dataframes=dfs)
    fits = poly_ellip_fits(deg=9, theta_range=spread, theta_step=step)

    # # Extract coefficients for 0 degrees
    # c1_coeffs = fits.loc[0, 'C1_fit']
    # c4_coeffs = fits.loc[0, 'C4_fit']

    # # Format the polynomial equations
    # c1_eq = format_poly(c1_coeffs)
    # c4_eq = format_poly(c4_coeffs)

    # print(f"Equation for C1 at 0 degrees: {c1_eq}")
    # print(f"Equation for C4 at 0 degrees: {c4_eq}")

if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("exited")



