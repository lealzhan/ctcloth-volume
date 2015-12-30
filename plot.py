import matplotlib.pylab as plt

def plotImage(img, cm=plt.cm.gray):
    fig, ax = plt.subplots()
    pic = ax.imshow(img, cmap=cm)
    fig.colorbar(pic)
    fig.show()


def plotHistogram(data, bins):
    hist, bins = np.histogram(data, bins=bins)    
    fig, ax = plt.subplots()
    
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align='center', width=width)  
    
    fig.show()

def plotSurface(x, y, z):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    fig.show()


if __name__ == '__main__':
    pass
