def visualize(image, nw, se):
    im = np.array(Image.open(image), dtype=np.uint8)
    fig,ax = plt.subplots(1)
    ax.imshow(im)
    (x_nw, y_nw) = nw
    (x_se, y_se) = se
    assert x_se >= x_nw
    assert y_nw <= y_se
    width = x_nw - x_se + 1
    height = y_nw - y_se + 1
    x = x_se
    y = y_se
    rect = patches.Rectangle((x, y),width,height,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

def main():
    image_path = 'stinkbug.png'
    visualize(image_path, (4, 4), (50, 50))
if __name__ == '__main__':
    main()
