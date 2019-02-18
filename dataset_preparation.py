import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix


def get_data_frame():
    # Load dataset
    # url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    # names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # dataset = pandas.read_csv(url, names=names)

    url = "iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # dataset = pandas.read_csv(url, names=names)
    return pandas.read_csv(url, names=names)


def main():
    dataset = get_data_frame();
    # shape
    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    print(dataset.groupby('class').size())

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    # histograms
    dataset.hist()
    plt.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()


if __name__ == '__main__':
    main()
