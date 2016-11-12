#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class MPlotHelper(object):

  def __init__(self, yinv=False):
    r""" init MPlotHelper 
	 by default matplotlib bottom left corner is (0,0)
	 it is possible to use top left corner as (0,0) by setting yinv to True"""
    self.yinv = yinv

  def figure(self, width, height, dpi):
    r""" Returns a matplotlib.figure.Figure instance of the given size (in pixels)"""
    self.width, self.height = float(width), float(height)
    # force a dpi=100 to work internally
    self.fig = plt.figure(figsize=(self.width/dpi,self.height/dpi), dpi=dpi)
    self.fig.canvas.mpl_connect('pick_event', self.onpick)
    self.subfigures = {}
    return self.fig

  def subfigure(self, pos, size, **extra):
    r""" Returns a new subfigure (axes object) with left bottom corner at the given position (in pixels) and of the given size (in pixels)"""
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=self.w(pos[0]), right=self.w(pos[0] + size[0]),
	      top=self.h(pos[1] + (-1)**(self.yinv) * size[1]), bottom=self.h(pos[1]))
    ax = self.fig.add_subplot(gs[0], **extra)
    self.current_figure = ax
    self.subfigures[ax] = { 'pos' : pos, 'width' : float(size[0]), 'height' : float(size[1]) } 
    self.current_figure_pos = pos
    self.current_figure_width = float(size[0])
    self.current_figure_height = float(size[1])
    return ax

  def cfp(self):
    r""" Returns current figure position"""
    return self.current_figure_pos

  def cf(self):
    r""" Returns current figure position, width and height"""
    return self.current_figure_pos, self.current_figure_width, self.current_figure_height

  def ar(self, pixels, maximum):
    r""" Returns the relative position equivalent to the given absolute position (in pixels) relative to the given maximum absolute position (in pixels)"""
    return float(pixels) / maximum
  
  def cw(self, pixels):
    r""" Returns the relative position equivalent to the given absolute position (in pixels) relative to the current subfigure width"""
    return self.ar(pixels, self.current_figure_width)
  def ch(self, pixels):
    r""" Returns the relative position equivalent to the given absolute position (in pixels) relative to the current subfigure height"""
    return self.ar(self.current_figure_height - pixels if self.yinv else pixels, self.current_figure_height)
  def c(self, width, height):
    r""" Returns the relative coordinates equivalent to the given absolute coordinates (in pixels) relative to the current subfigure"""
    return self.cw(width), self.ch(height)

  def w(self, pixels):
    r""" Returns the relative position equivalent to the given absolute position (in pixels) relative to the current figure width"""
    return self.ar(pixels, self.width)
  def h(self, pixels):
    r""" Returns the relative position equivalent to the given absolute position (in pixels) relative to the current figure height"""
    return self.ar(self.height - pixels if self.yinv else pixels, self.height)
  def f(self, width, height):
    r""" Returns the relative coordinates equivalent to the given absolute coordinates (in pixels) relative to the current figure"""
    return self.w(width), self.h(height)

  def clegend(self, **extra):
    r""" Print the legend of the current subfigure"""
    leg = self.current_figure.legend(**extra)
    self.current_figure_legend = {}
    sf = self.subfigures[self.current_figure] 
    sf['legend'] = {}
    for label, line in zip(leg.get_lines(), self.current_figure.get_lines()):
      label.set_picker(5)  # 5 pts tolerance
      sf['legend'][label] = line

  def onpick(self, event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    label = event.artist
    ax = event.mouseevent.inaxes
    line = self.subfigures[ax]['legend'][label]
    vis = not line.get_visible()
    line.set_visible(vis)
    label.set_alpha(1.0 if vis else 0.2) # change label alpha given visibility
    self.fig.canvas.draw()

# if __name__ == '__main__':
#   # example of use
#   from pylab import *

  ### Using the default version (0,0) is the bottom left corner
  # p = MPlotHelper()
  # # define a figure with a width of 1600px and height of 600px
  # #p.figure(1600, 600)
  # # define a subfigure with left bottom point at (100px,100px) with width of 250px and height of 400px
  # ax1 = p.subfigure((100,100), (250, 400))
  # ax1.plot(arange(0,10,1),randn(10),label='a')
  # ax1.plot(arange(0,10,1),randn(10),label='b')
  # p.clegend()
  # # set a title at position (0px,400px) of the current subfigure
  # ax1.set_title("O", position=p.c(0, 400), weight='bold')
  # # define a subfigure with left bottom point at (450px,100px) with width of 250px and height of 400px 
  # ax2 = p.subfigure((450,100), (150, 400), sharey=ax1)
  # ax2.plot(arange(0,15,1),randn(15) * 2)
  # # set a title at position (-50px,400px) of the current subfigure
  # ax2.set_title("X", position=p.c(-50, 400), weight='bold')
  # # define a subfigure with left bottom point at (800px,100px) with width of 250px and height of 200px
  # ax3 = p.subfigure((800,100), (250, 200))
  # ax3.plot(arange(0,10,1),randn(10))
  # # define a subfigure with left bottom point at (900px,200px) with width of 250px and height of 200px
  # ax4 = p.subfigure((900,200), (250, 200))
  # ax4.patch.set_alpha(0.5) # to make ax4 background semi transparent
  # ax4.plot(arange(0,10,1),randn(10), label='c')
  # p.clegend()
  # # set a title at position (800px,50px) of the global figure
  # plt.figtext(p.w(800), p.h(50), "A", ha='center', va='top')
  # # use what dpi you want here
  # # plt.savefig("test.png", dpi=200)
  # plt.show()
