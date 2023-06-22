#Importowanie niezbędnych bibliotek: TensorFlow, NumPy i matplotlib. Wyświetlanie wersji TensorFlow.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
print('Wersja:')
print(tf.__version__)
fashion_mnist = tf.keras.datasets.fashion_mnist


#Załadowanie zestawu danych Fashion MNIST, który zawiera obrazy ubrań oraz odpowiadające im etykiety. 
#Zestaw danych zostaje podzielony na dane treningowe (train_images, train_labels) i dane testowe (test_images, test_labels).
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#Lista z nazwami klas dla etykiet danych. Odpowiednie etykiety są przypisane do każdej klasy od 0 do 9.
#class_names = ['zero', 'jeden', 'dwa', 'trzy', 'cztery',
#'pięć', 'sześć', 'siedem', 'osiem', 'dziewięć']

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Wyświetlanie rozmiaru i liczby obiektów treningowych. Train_images to trójwymiarowa macierz zawierająca obrazy, 
#a train_labels to jednowymiarowa macierz zawierająca odpowiadające im etykiety.
print('Rozmiar i ilość obiektów treningowych:')
print(train_images.shape)
print(train_labels)


#Wyświetlanie rozmiaru i liczby obiektów testowych. Test_images to trójwymiarowa macierz zawierająca obrazy testowe, 
#a test_labels to jednowymiarowa macierz zawierająca odpowiadające im etykiety.
print('Rozmiar i ilość obiektów testowych:')
print(test_images.shape)
print(test_labels)


#Wyświetlenie pierwszego obrazu treningowego z zestawu danych. Opcja colorbar() wyświetla pasek kolorów dla wartości pikseli. 
#Opcja grid(False) usuwa siatkę na wykresie.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


#Normalizacja wartości pikseli obrazów poprzez podzielenie ich przez 255. Dzięki temu wartości pikseli mieszczą się w zakresie od 0 do 1.
train_images = train_images / 255.
test_images = test_images / 255.0


#Wyświetlenie ponowne pierwszego obrazu treningowego po normalizacji.
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


#Wyświetlenie 25 obrazów treningowych w siatce 5x5. Każdy obraz jest etykietowany i wyświetlany w odcieniach szarości.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#Definicja sekwencyjnego modelu sieci neuronowej w TensorFlow. Model składa się z warstwy wejściowej (Flatten), 
#warstwy ukrytej z 128 neuronami (Dense) i warstwy wyjściowej z 10 neuronami, odpowiadającymi 10 klasom.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


#Skompilowanie modelu. Określenie optymalizatora (Adam), funkcji straty (SparseCategoricalCrossentropy) i metryki (accuracy) do oceny wydajności modelu.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#Trenowanie modelu na danych treningowych. Określenie liczby epok (10) i danych walidacyjnych dla oceny wydajności podczas treningu.
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


#Wykresowanie wykresu dokładności modelu na zbiorze treningowym i walidacyjnym w zależności od liczby epok. Oś x reprezentuje epoki, a oś y reprezentuje dokładność. 
#Wykres ma etykiety osi oraz legendę.
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


#Ewaluacja modelu na danych testowych. Obliczenie straty i dokładności na podstawie danych testowych.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)


#Wyświetlenie dokładności modelu na danych testowych.
print('\nTest accuracy:', test_acc)


#Utworzenie nowego modelu, który zawiera oryginalny model oraz warstwę Softmax na końcu. Wykonanie predykcji na danych testowych za pomocą modelu. 
#Wyświetlenie wyników predykcji dla pierwszego obrazu testowego, w tym prawdopodobieństw przynależności do poszczególnych klas, indeksu klasy o najwyższym prawdopodobieństwie i rzeczywistej etykiety.
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


#Wyświetlenie podsumowania architektury modelu, w tym informacji o kształcie danych na wejściu i wyjściu każdej warstwy.
print(model.summary())


#Definicja funkcji pomocniczych do wyświetlania obrazów testowych wraz z ich predykcjami i prawdopodobieństwami przynależności do klas.
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


#Wyświetlenie kilku obrazów testowych wraz z ich predykcjami i prawdopodobieństwami przynależności do klas. 
#Obrazy są wyświetlane w siatce o rozmiarze num_rows na num_cols, a dla każdego obrazu wyświetlane są dwie podfigury - obraz z predykcją i wykres słupkowy przedstawiający prawdopodobieństwa.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
