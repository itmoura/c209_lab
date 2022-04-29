#!/usr/bin/env python


DEBUG = True

#funcao q mostra as imagens
if DEBUG:
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)


DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'mnist'  # `'mnist'` or `'fashion-mnist'`
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images.idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images.idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels.idx1-ubyte'

#transforma bytes em inteiros
def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

#funcao q le os arquivos que contem os numeros e usa para treinar a IA
def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4) #bytes q definem o tipo do arquivo
        n_images = bytes_to_int(f.read(4)) #numero de imagens dentro do arquivo
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4)) #numero de linhas
        n_columns = bytes_to_int(f.read(4)) #numero de colunas
        for image_idx in range(n_images): #iteracao q cria as matrizes de pixel no formato dos numeros
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images

#funcao q diz qual é o numero de 0 a 9 q o arquivo do dataset é pra representar
def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels

#rearranja as informaçoes da matriz de pixels em um vetor unidimensional
def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]

#cria um vetor a partir das inforamções da imagem
def extract_features(X):
    return [flatten_list(sample) for sample in X]

#funcao para o calculo de distancia euclidiana entre os vetores x e y
def dist(x, y):
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)

#funcao usada em knn pra pegar a distancia entre os dados
def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)

#k nearest neighbor
#compara o numero com todos os outros numeros, e pega os que são mais próximos
def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_sample_idx, test_sample in enumerate(X_test): #para cada caso de teste
        print(test_sample_idx, end=' ', flush=True)
        training_distances = get_training_distances_for_test_sample( #calcula a distancia a partir da origem
            X_train, test_sample
        )
        sorted_distance_indices = [ #encontrando os caso mais próximos
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]   #dos numeros que estão mais próximos, pega o valor que aparece mais frequentemente
        top_candidate = get_most_frequent_element(candidates) 
        y_pred.append(top_candidate)
    print()
    return y_pred


def get_garment_from_label(label):
    return [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot',
    ][label]


def main():
    n_train = 1000
    n_test = 10
    k = 7
    print(f'Dataset: {DATASET}')
    print(f'n_train: {n_train}')
    print(f'n_test: {n_test}')
    print(f'k: {k}')
    X_train = read_images(TRAIN_DATA_FILENAME, n_train) #imagens que sabemos quais numeros representam
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    X_test = read_images(TEST_DATA_FILENAME, n_test) #numeros q nao sabemos e queremos q o algoritmo adivinhe
    y_test = read_labels(TEST_LABELS_FILENAME, n_test)

    if DEBUG:
        #Salvando as imagens para poder visualizar
        for idx, test_sample in enumerate(X_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test, k)

    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_pred, y_test)
    ]) / len(y_test)

    if DATASET == 'fashion-mnist':
        garments_pred = [
            get_garment_from_label(label)
            for label in y_pred
        ]
        print(f'Predicted garments: {garments_pred}')
    else:
        print(f'Predicted labels: {y_pred}')

    print(f'Accuracy: {accuracy * 100}%')


if __name__ == '__main__':
    main()
