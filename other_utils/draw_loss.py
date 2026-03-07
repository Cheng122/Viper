"""
Visualization utilities for training loss curves.

This module provides functions to plot and save training loss curves.
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def draw_loss_curve(loss_history, save_path, title='Training Loss'):
    """
    Draw and save loss curve.
    
    Args:
        loss_history: List of loss values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved loss curve to {save_path}')


def draw_multi_loss_curves(loss_dict, save_path, title='Training Loss'):
    """
    Draw and save multiple loss curves on the same plot.
    
    Args:
        loss_dict: Dictionary of loss name -> loss history list
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(12, 8))
    for name, history in loss_dict.items():
        plt.plot(history, label=name)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved loss curves to {save_path}')


def draw_success_rate(success_history, save_path, title='Success Rate'):
    """
    Draw and save success rate curve.
    
    Args:
        success_history: List of success rate values
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(success_history)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Success Rate')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f'Saved success rate curve to {save_path}')


if __name__ == '__main__':
    example_loss = [1.0 - 0.001 * i for i in range(100)]
    draw_loss_curve(example_loss, 'example_loss.png')
