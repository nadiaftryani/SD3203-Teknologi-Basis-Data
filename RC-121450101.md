Why would you want to know more about different ways of storing and accessing images in Python? If you’re [segmenting a handful of images by color](https://realpython.com/python-opencv-color-spaces/) or [detecting faces one by one](https://realpython.com/face-recognition-with-python/) using OpenCV, then you don’t need to worry about it. Even if you’re using the [Python Imaging Library (PIL)](https://realpython.com/image-processing-with-the-python-pillow-library/) to draw on a few hundred photos, you still don’t need to. Storing images on disk, as `.png` or `.jpg` files, is both suitable and appropriate.

Increasingly, however, the number of images required for a given task is getting larger and larger. Algorithms like convolutional neural networks, also known as convnets or CNNs, can handle enormous datasets of images and even learn from them. If you’re interested, you can read more about how convnets can be used for [ranking selfies](https://karpathy.github.io/2015/10/25/selfie/) or for [sentiment analysis](https://realpython.com/sentiment-analysis-python/).

[ImageNet](http://image-net.org/) is a well-known public image database put together for training models on tasks like object classification, detection, and segmentation, and it consists of _over 14 million images._

Think about how long it would take to load all of them into memory for training, in batches, perhaps hundreds or thousands of times. Keep reading, and you’ll be convinced that it would take quite awhile—at least long enough to leave your computer and do many other things while you wish you worked at Google or NVIDIA.

**In this tutorial, you’ll learn about:**

-   Storing images on disk as `.png` files
-   Storing images in lightning memory-mapped databases (LMDB)
-   Storing images in hierarchical data format (HDF5)

**You’ll also explore the following:**

-   Why alternate storage methods are worth considering
-   What the performance differences are when you’re reading and writing single images
-   What the performance differences are when you’re reading and writing _many_ images
-   How the three methods compare in terms of disk usage

If none of the storage methods ring a bell, don’t worry: for this article, all you need is a reasonably solid foundation in Python and a basic understanding of images (that they are really composed of multi-dimensional arrays of [numbers](https://realpython.com/python-numbers/)) and relative memory, such as the difference between 10MB and 10GB.

Let’s get started!

## Setup

You will need an image dataset to experiment with, as well as a few [Python packages](https://realpython.com/python-modules-packages/).

### A Dataset to Play With

We will be using the Canadian Institute for Advanced Research image dataset, better known as [CIFAR-10](https://en.wikipedia.org/wiki/CIFAR-10), which consists of 60,000 32x32 pixel color images belonging to different object classes, such as dogs, cats, and airplanes. Relatively, CIFAR is not a very large dataset, but if we were to use the full [TinyImages dataset](https://groups.csail.mit.edu/vision/TinyImages/), then you would need about 400GB of free disk space, which would probably be a limiting factor.

Credits for the dataset as described in [chapter 3 of this tech report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) go to Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.

If you’d like to follow along with the code examples in this article, you can [download CIFAR-10 here](https://www.cs.toronto.edu/~kriz/cifar.html), selecting the Python version. You’ll be sacrificing 163MB of disk space:

[![cifar-10-dataset](https://files.realpython.com/media/cifar_10.e77ef0cd86df.png)](https://files.realpython.com/media/cifar_10.e77ef0cd86df.png)

Image: A. Krizhevsky

When you download and unzip the folder, you’ll discover that the files are not human-readable image files. They have actually been serialized and saved in batches using [cPickle](https://docs.python.org/2/library/pickle.html).

While we won’t consider [`pickle`](https://realpython.com/python-pickle-module/) or `cPickle` in this article, other than to extract the CIFAR dataset, it’s worth mentioning that the Python `pickle` module has the key advantage of being able to serialize any Python object without any extra code or transformation on your part. It also has a potentially serious disadvantage of posing a security risk and not coping well when dealing with very large quantities of data.

The following code unpickles each of the five batch files and loads all of the images into a NumPy array:

All the images are now in RAM in the `images` [variable](https://realpython.com/python-variables/), with their corresponding meta data in `labels`, and are ready for you to manipulate. Next, you can install the Python packages you’ll use for the three methods.

### Setup for Storing Images on Disk

Anda harus menyiapkan lingkungan Anda untuk metode default menyimpan dan mengakses gambar-gambar ini dari disk. Artikel ini akan mengasumsikan Anda telah menginstal Python 3.x di sistem Anda, dan akan digunakan `Pillow`untuk manipulasi gambar:

Alternatifnya, jika mau, Anda dapat menginstalnya menggunakan [Anaconda](https://anaconda.org/conda-forge/pillow) :

Sekarang Anda siap untuk menyimpan dan membaca gambar dari disk.

### Memulai Dengan LMDB

[LMDB](https://en.wikipedia.org/wiki/Lightning_Memory-Mapped_Database) , terkadang disebut sebagai “Lightning Database,” adalah singkatan dari Lightning Memory-Mapped Database karena cepat dan menggunakan file yang dipetakan memori. Ini adalah penyimpanan nilai kunci, bukan database relasional.

Dalam hal implementasi, LMDB adalah pohon B+, yang pada dasarnya berarti struktur grafik mirip pohon yang disimpan dalam memori di mana setiap elemen nilai kunci adalah sebuah simpul, dan simpul dapat memiliki banyak anak. Node pada level yang sama dihubungkan satu sama lain untuk traversal yang cepat.

Yang terpenting, komponen kunci pohon B+ diatur agar sesuai dengan ukuran halaman sistem operasi host, memaksimalkan efisiensi saat mengakses pasangan nilai kunci apa pun dalam database. Karena kinerja tinggi LMDB sangat bergantung pada poin khusus ini, efisiensi LMDB terbukti bergantung pada sistem file yang mendasarinya dan implementasinya.

Alasan utama lainnya mengapa LMDB efisien adalah karena LMDB dipetakan dalam memori. Artinya, **ia mengembalikan penunjuk langsung ke alamat memori dari kunci dan nilai** , tanpa perlu menyalin apa pun di memori seperti yang dilakukan kebanyakan database lainnya.

Mereka yang ingin mendalami lebih dalam tentang detail implementasi internal pohon B+ dapat membaca [artikel tentang pohon B+ ini](http://www.cburch.com/cs/340/reading/btree/index.html) dan kemudian bermain dengan [visualisasi penyisipan simpul ini](https://www.cs.usfca.edu/~galles/visualization/BPlusTree.html) .

Jika pohon B+ tidak menarik minat Anda, jangan khawatir. Anda tidak perlu tahu banyak tentang implementasi internalnya untuk menggunakan LMDB. Kami akan menggunakan [pengikatan Python](https://realpython.com/python-bindings-overview/) untuk perpustakaan LMDB C, yang dapat diinstal melalui pip:

Anda juga memiliki opsi untuk menginstal melalui [Anaconda](https://anaconda.org/conda-forge/python-lmdb) :

Periksa apakah Anda dapat melakukannya `import lmdb`dari shell Python, dan Anda siap melakukannya.

### Memulai Dengan HDF5

HDF5 adalah singkatan dari Hierarchical Data Format, format file yang disebut sebagai HDF4 atau HDF5. Kita tidak perlu khawatir tentang HDF4, karena HDF5 adalah versi yang dipertahankan saat ini.

Menariknya, HDF [berasal](https://www.hdfgroup.org/about-us/) dari National Center for Supercomputing Applications, sebagai format data ilmiah yang portabel dan ringkas. Jika Anda bertanya-tanya apakah ini digunakan secara luas, lihat [uraian singkat NASA tentang HDF5](https://earthdata.nasa.gov/user-resources/standards-and-references/hdf-eos5) dari proyek Data Bumi mereka.

File HDF terdiri dari dua jenis objek:

1.  Kumpulan data
2.  Grup

Kumpulan data adalah array multidimensi, dan grup terdiri dari kumpulan data atau grup lain. Array multidimensi dengan ukuran dan tipe apa pun dapat disimpan sebagai kumpulan data, namun dimensi dan tipenya harus seragam dalam kumpulan data. Setiap dataset harus berisi array berdimensi N yang homogen. Meskipun demikian, karena grup dan kumpulan data mungkin disarangkan, Anda masih bisa mendapatkan heterogenitas yang mungkin Anda perlukan:

Seperti perpustakaan lainnya, Anda dapat menginstal secara bergantian melalui [Anaconda](https://anaconda.org/conda-forge/h5py) :

Jika Anda bisa `import h5py`dari shell Python, semuanya sudah diatur dengan benar.

## Menyimpan Satu Gambar

Sekarang setelah Anda memiliki gambaran umum tentang metode ini, mari selami dan lihat perbandingan kuantitatif dari tugas-tugas dasar yang penting bagi kami: **berapa lama waktu yang dibutuhkan untuk membaca dan menulis file, dan berapa banyak memori disk yang akan digunakan.** Ini juga akan berfungsi sebagai pengenalan dasar tentang cara kerja metode, dengan contoh kode cara menggunakannya.

Saat saya mengacu pada "file", yang saya maksud biasanya banyak. Namun, penting untuk membedakannya karena beberapa metode mungkin dioptimalkan untuk operasi dan jumlah file yang berbeda.

Untuk keperluan eksperimen, **kita dapat membandingkan kinerja antara berbagai jumlah file, dengan faktor 10 dari satu gambar hingga 100.000 gambar.** Karena lima kumpulan CIFAR-10 kami berjumlah 50.000 gambar, kami dapat menggunakan setiap gambar dua kali untuk mendapatkan 100.000 gambar.

Untuk mempersiapkan percobaan, Anda perlu membuat folder untuk setiap metode, yang akan berisi semua file atau gambar database, dan menyimpan jalur ke direktori tersebut dalam variabel:

`Path`tidak secara otomatis membuat folder untuk Anda kecuali Anda secara khusus memintanya untuk:

Sekarang Anda dapat melanjutkan menjalankan eksperimen sebenarnya, dengan contoh kode tentang cara melakukan tugas dasar dengan tiga metode berbeda. Kita dapat menggunakan `timeit`modul, yang disertakan dalam pustaka standar Python, untuk membantu mengatur waktu eksperimen.

Meskipun tujuan utama artikel ini bukan untuk mempelajari API dari berbagai paket Python, akan sangat membantu jika kita memahami bagaimana paket tersebut dapat diimplementasikan. Kami akan membahas prinsip umum beserta semua kode yang digunakan untuk melakukan eksperimen penyimpanan.

### Menyimpan ke Disk

Masukan kami untuk percobaan ini adalah satu gambar `image`, yang saat ini ada di memori sebagai array NumPy. Anda ingin menyimpannya terlebih dahulu ke disk sebagai `.png`image, dan menamainya menggunakan image ID yang unik `image_id`. Ini dapat dilakukan dengan menggunakan `Pillow`paket yang Anda instal sebelumnya:

Ini akan menyimpan gambar. Dalam semua aplikasi realistis, Anda juga peduli dengan meta data yang dilampirkan pada gambar, yang dalam contoh kumpulan data kami adalah label gambar. Saat Anda menyimpan gambar ke disk, ada beberapa opsi untuk menyimpan data meta.

Salah satu solusinya adalah dengan mengkodekan label ke dalam nama gambar. Keuntungannya adalah tidak memerlukan file tambahan apa pun.

Namun, ini juga memiliki kelemahan besar karena memaksa Anda menangani semua file setiap kali Anda melakukan sesuatu dengan label. Menyimpan label dalam file terpisah memungkinkan Anda bermain-main dengan label saja, tanpa harus memuat gambar. Di atas, saya telah menyimpan label dalam `.csv`file terpisah untuk percobaan ini.

Sekarang mari kita lanjutkan melakukan tugas yang sama persis dengan LMDB.

### Menyimpan ke LMDB

Pertama, LMDB adalah sistem penyimpanan nilai kunci di mana setiap entri disimpan sebagai array byte, jadi dalam kasus kita, kunci akan menjadi pengidentifikasi unik untuk setiap gambar, dan nilainya akan menjadi gambar itu sendiri. **Baik kunci maupun nilai diharapkan berupa strings** , jadi penggunaan umum adalah membuat serial nilai sebagai string, lalu membatalkan serialisasinya saat membacanya kembali.

Anda dapat menggunakannya `pickle`untuk serialisasi. Objek Python apa pun dapat dibuat serial, jadi sebaiknya Anda juga menyertakan data meta gambar ke dalam database. Ini menyelamatkan Anda dari kesulitan melampirkan meta data kembali ke data gambar saat kami memuat kumpulan data dari disk.

Anda dapat membuat [kelas Python](https://realpython.com/python-classes/) dasar untuk gambar dan meta datanya:

Kedua, karena LMDB dipetakan dengan memori, database baru perlu mengetahui berapa banyak memori yang diperkirakan akan digunakan. Hal ini relatif mudah dalam kasus kami, namun dapat menjadi masalah besar dalam kasus lain, yang akan Anda lihat lebih mendalam di bagian selanjutnya. LMDB menyebut variabel ini sebagai `map_size`.

Terakhir, operasi baca dan tulis dengan LMDB dilakukan di `transactions`. Anda dapat menganggapnya mirip dengan database tradisional, yang terdiri dari sekelompok operasi pada database. Ini mungkin terlihat jauh lebih rumit daripada versi disk, tapi tunggu dulu dan teruslah membaca!

Dengan mengingat tiga poin tersebut, mari kita lihat kode untuk menyimpan satu gambar ke LMDB:

Anda sekarang siap menyimpan gambar ke LMDB. Terakhir, mari kita lihat metode terakhir, HDF5.

### Menyimpan Dengan HDF5

Ingatlah bahwa file HDF5 dapat berisi lebih dari satu kumpulan data. Dalam kasus yang agak sepele ini, Anda dapat membuat dua himpunan data, satu untuk gambar, dan satu lagi untuk meta datanya:

`h5py.h5t.STD_U8BE`menentukan jenis data yang akan disimpan dalam kumpulan data, yang dalam hal ini adalah bilangan bulat 8-bit yang tidak ditandatangani. Anda dapat melihat [daftar lengkap tipe data HDF yang telah ditentukan sebelumnya di sini](http://api.h5py.org/h5t.html) .

Sekarang kita telah meninjau tiga metode menyimpan satu gambar, mari kita lanjutkan ke langkah berikutnya.

### Eksperimen untuk Menyimpan Satu Gambar

Sekarang Anda dapat memasukkan ketiga fungsi untuk menyimpan satu gambar ke dalam [kamus](https://realpython.com/python-dicts/) , yang dapat dipanggil nanti selama eksperimen pengaturan waktu:

Akhirnya, semuanya siap untuk melakukan percobaan waktunya. Mari kita coba menyimpan gambar pertama dari CIFAR dan label terkait, lalu menyimpannya dengan tiga cara berbeda:

Ingatlah bahwa kami tertarik pada waktu proses, yang ditampilkan di sini dalam hitungan detik, dan juga penggunaan memori:

| metode | Simpan Gambar Tunggal + Meta | Penyimpanan |
| --- | --- | --- |
| Disk | 1,915 mdtk | 8K |
| LMDB | 1,203 ms | 32 K |
| HDF5 | 8,243 ms | 8K |

Ada dua kesimpulan di sini:

1.  Semua metode ini cukup cepat.
2.  Dalam hal penggunaan disk, LMDB menggunakan lebih banyak.

Jelasnya, meskipun LMDB memiliki sedikit keunggulan kinerja, kami belum meyakinkan siapa pun mengapa tidak menyimpan gambar saja di disk. Bagaimanapun, ini adalah format yang dapat dibaca manusia, dan Anda dapat membuka dan melihatnya dari browser sistem file apa pun! Nah, sekarang saatnya melihat lebih banyak gambar…

## Menyimpan Banyak Gambar

Anda telah melihat kode untuk menggunakan berbagai metode penyimpanan untuk menyimpan satu gambar, jadi sekarang kita perlu menyesuaikan kode untuk menyimpan banyak gambar dan kemudian menjalankan eksperimen berwaktu.

### Menyesuaikan Kode untuk Banyak Gambar

Menyimpan _banyak_ gambar sebagai `.png`file semudah menelepon `store_single_method()`beberapa kali. Namun hal ini tidak berlaku untuk LMDB atau HDF5, karena Anda tidak ingin file database berbeda untuk setiap gambar. Sebaliknya, Anda ingin memasukkan semua gambar ke dalam satu atau lebih file.

Anda perlu sedikit mengubah kode dan membuat tiga fungsi baru yang menerima banyak gambar, `store_many_disk()`, `store_many_lmdb()`, dan `store_many_hdf5`:

Agar Anda dapat menyimpan lebih dari satu file ke disk, metode file gambar diubah untuk mengulang setiap gambar dalam daftar. Untuk LMDB, loop juga diperlukan karena kita membuat `CIFAR_Image`objek untuk setiap gambar dan meta datanya.

Penyesuaian terkecil adalah dengan metode HDF5. Faktanya, hampir tidak ada penyesuaian sama sekali! File HFD5 tidak memiliki batasan ukuran file selain batasan eksternal atau ukuran kumpulan data, sehingga semua gambar dimasukkan ke dalam satu kumpulan data, seperti sebelumnya.

Selanjutnya, Anda perlu menyiapkan kumpulan data untuk eksperimen dengan memperbesar ukurannya.

### Mempersiapkan Kumpulan Data

Sebelum menjalankan eksperimen lagi, pertama-tama mari gandakan ukuran kumpulan data kita sehingga kita dapat menguji hingga 100.000 gambar:

Sekarang gambar sudah cukup, saatnya bereksperimen.

### Eksperimen untuk Menyimpan Banyak Gambar

Seperti yang Anda lakukan saat membaca banyak gambar, Anda dapat membuat kamus yang menangani semua fungsi `store_many_`dan menjalankan eksperimen:

Jika Anda mengikuti dan menjalankan kodenya sendiri, Anda harus duduk santai sejenak dan menunggu hingga 111.110 gambar disimpan masing-masing tiga kali ke disk Anda, dalam tiga format berbeda. Anda juga harus mengucapkan selamat tinggal pada ruang disk sekitar 2 GB.

Sekarang untuk momen kebenaran! **Berapa lama waktu yang dibutuhkan untuk semua penyimpanan itu?** Sebuah gambar bernilai ribuan kata:

[![menyimpan banyak](https://files.realpython.com/media/store_many.273573157770.png)](https://files.realpython.com/media/store_many.273573157770.png)

[![simpan-banyak-log](https://files.realpython.com/media/store_many_log.29e8ae980ab6.png)](https://files.realpython.com/media/store_many_log.29e8ae980ab6.png)

Grafik pertama menunjukkan waktu penyimpanan normal dan tidak disesuaikan, menyoroti perbedaan drastis antara penyimpanan ke `.png`file dan LMDB atau HDF5.

Grafik kedua menunjukkan `log`perubahan waktu, menyoroti bahwa HDF5 dimulai lebih lambat dibandingkan LMDB tetapi, dengan jumlah gambar yang lebih besar, hasilnya sedikit lebih cepat.

Meskipun hasil pastinya mungkin berbeda-beda tergantung mesin Anda, **inilah alasan mengapa LMDB dan HDF5 layak untuk dipertimbangkan.** Berikut kode yang menghasilkan grafik di atas:

Sekarang mari kita lanjutkan membaca gambarnya kembali.

## Membaca Satu Gambar

Pertama, mari kita pertimbangkan kasus untuk membaca satu gambar kembali ke dalam array untuk masing-masing dari tiga metode.

### Membaca Dari Disk

Dari ketiga metode tersebut, LMDB memerlukan kerja keras paling banyak saat membaca kembali file gambar dari memori, karena langkah serialisasi. Mari kita telusuri fungsi-fungsi yang membaca satu gambar untuk masing-masing dari tiga format penyimpanan.

Pertama, baca satu gambar dan meta-nya dari file `.png`dan `.csv`:

### Membaca Dari LMDB

Selanjutnya, baca gambar dan meta yang sama dari LMDB dengan membuka lingkungan dan memulai transaksi baca:

Berikut beberapa hal yang tidak boleh dilakukan tentang cuplikan kode di atas:

-   **Baris 13:** Bendera `readonly=True`menentukan bahwa tidak ada penulisan yang diperbolehkan pada file LMDB sampai transaksi selesai. Dalam istilah basis data, ini setara dengan mengambil kunci baca.
-   **Baris 20:** Untuk mengambil objek CIFAR\_Image, Anda perlu membalik langkah yang kita ambil untuk mengambilnya saat kita menulisnya. Di sinilah fungsi `get_image()`objek berguna.

Ini mengakhiri pembacaan kembali gambar dari LMDB. Terakhir, Anda ingin melakukan hal yang sama dengan HDF5.

### Membaca Dari HDF5

Membaca dari HDF5 terlihat sangat mirip dengan proses menulis. Berikut adalah kode untuk membuka dan membaca file HDF5 serta mengurai gambar dan meta yang sama:

Perhatikan bahwa Anda mengakses berbagai kumpulan data dalam file dengan mengindeks `file`objek menggunakan nama kumpulan data yang diawali dengan garis miring `/`. Seperti sebelumnya, Anda dapat membuat kamus yang berisi semua fungsi baca:

Setelah kamus ini disiapkan, Anda siap menjalankan eksperimen.

### Eksperimen untuk Membaca Satu Gambar

Anda mungkin berharap bahwa eksperimen untuk membaca satu gambar akan memberikan hasil yang agak sepele, namun berikut kode eksperimennya:

Berikut hasil percobaan membaca satu gambar:

| metode | Baca Gambar Tunggal + Meta |
| --- | --- |
| Disk | 1,61970 mdtk |
| LMDB | 4,52063 mdtk |
| HDF5 | 1,98036 mdtk |

Ini sedikit lebih cepat untuk membaca `.png`dan `.csv`file langsung dari disk, tetapi ketiga metode tersebut bekerja dengan sangat cepat. Eksperimen yang akan kita lakukan selanjutnya jauh lebih menarik.

## Membaca Banyak Gambar

Sekarang Anda dapat menyesuaikan kode untuk membaca banyak gambar sekaligus. Ini mungkin tindakan yang paling sering Anda lakukan, jadi performa runtime sangat penting.

### Menyesuaikan Kode untuk Banyak Gambar

Dengan memperluas fungsi di atas, Anda dapat membuat fungsi dengan `read_many_`, yang dapat digunakan untuk percobaan berikutnya. Seperti sebelumnya, menarik untuk membandingkan kinerja saat membaca jumlah gambar yang berbeda, yang diulangi dalam kode di bawah ini untuk referensi:

Dengan fungsi membaca yang disimpan dalam kamus seperti halnya fungsi menulis, Anda siap untuk bereksperimen.

### Eksperimen Membaca Banyak Gambar

Anda sekarang dapat menjalankan eksperimen untuk membaca banyak gambar:

Seperti yang kami lakukan sebelumnya, Anda dapat membuat grafik hasil eksperimen yang telah dibaca:

[![baca-banyak-gambar](https://files.realpython.com/media/read_many.9c4a6dc5bdc0.png)](https://files.realpython.com/media/read_many.9c4a6dc5bdc0.png)

[![baca-banyak-log](https://files.realpython.com/media/read_many_log.594dac8746ad.png)](https://files.realpython.com/media/read_many_log.594dac8746ad.png)

Grafik atas menunjukkan waktu baca normal dan tidak disesuaikan, menunjukkan perbedaan drastis antara membaca dari `.png`file dan LMDB atau HDF5.

Sebaliknya, grafik di bawah menunjukkan variasi `log`waktu, menyoroti perbedaan relatif dengan gambar yang lebih sedikit. Yaitu, kita dapat melihat bagaimana HDF5 dimulai dari belakang, namun dengan lebih banyak gambar, secara konsisten menjadi lebih cepat dibandingkan LMDB dengan selisih yang kecil.

**Dalam praktiknya, waktu menulis seringkali kurang penting dibandingkan waktu membaca.** Bayangkan Anda sedang melatih jaringan neural dalam pada gambar, dan hanya setengah dari seluruh kumpulan data gambar Anda yang dapat dimasukkan ke dalam RAM sekaligus. Setiap periode pelatihan jaringan memerlukan seluruh kumpulan data, dan model memerlukan beberapa ratus periode untuk menyatu. Anda pada dasarnya akan membaca setengah dari kumpulan data ke dalam memori setiap zaman.

Ada beberapa trik yang dilakukan orang, seperti melatih zaman semu untuk menjadikannya sedikit lebih baik, tetapi Anda mengerti maksudnya.

Sekarang, lihat kembali grafik yang telah dibaca di atas. Perbedaan antara waktu baca 40 detik dan 4 detik secara tiba-tiba adalah perbedaan antara menunggu enam jam hingga model Anda dilatih, atau empat puluh menit!

Jika kita melihat waktu baca dan tulis pada grafik yang sama, kita mendapatkan yang berikut:

[![Baca tulis](https://files.realpython.com/media/read_write.a4f87d39489d.png)](https://files.realpython.com/media/read_write.a4f87d39489d.png)

Saat Anda menyimpan gambar sebagai `.png`file, ada perbedaan besar antara waktu tulis dan baca. Namun, dengan LMDB dan HDF5, perbedaannya tidak terlalu mencolok. Secara keseluruhan, meskipun waktu baca lebih penting daripada waktu tulis, terdapat argumen kuat untuk menyimpan gambar menggunakan LMDB atau HDF5.

Sekarang setelah Anda melihat manfaat kinerja LMDB dan HDF5, mari kita lihat metrik penting lainnya: penggunaan disk.

## Mempertimbangkan Penggunaan Disk

Kecepatan bukan satu-satunya metrik kinerja yang mungkin Anda minati. Kita sudah menangani kumpulan data yang sangat besar, jadi ruang disk juga merupakan masalah yang sangat valid dan relevan.

Misalkan Anda memiliki kumpulan data gambar sebesar 3 TB. Agaknya, Anda sudah menyimpannya di disk di suatu tempat, tidak seperti contoh CIFAR kami, jadi dengan menggunakan metode penyimpanan alternatif, Anda pada dasarnya membuat salinannya, yang juga harus disimpan. Melakukan hal ini akan memberi Anda manfaat kinerja yang besar saat Anda menggunakan gambar, namun Anda harus memastikan Anda memiliki cukup ruang disk.

**Berapa banyak ruang disk yang digunakan berbagai metode penyimpanan?** Berikut ruang disk yang digunakan setiap metode untuk setiap jumlah gambar:

[![toko-mem-gambar](https://files.realpython.com/media/store_mem.dadff24e67e3.png)](https://files.realpython.com/media/store_mem.dadff24e67e3.png)

Baik HDF5 dan LMDB menggunakan lebih banyak ruang disk dibandingkan jika Anda menyimpan menggunakan `.png`gambar normal. Penting untuk dicatat bahwa penggunaan dan kinerja disk LMDB dan HDF5 **sangat bergantung pada berbagai faktor, termasuk sistem operasi dan, yang lebih penting, ukuran data yang Anda simpan.**

LMDB memperoleh efisiensinya dari caching dan memanfaatkan ukuran halaman OS. Anda tidak perlu memahami cara kerja bagian dalamnya, namun perhatikan bahwa **dengan gambar yang lebih besar, Anda akan mendapatkan penggunaan disk yang jauh lebih banyak dengan LMDB,** karena gambar tidak akan muat di halaman daun LMDB, lokasi penyimpanan reguler di pohon, dan sebaliknya Anda akan memiliki banyak halaman yang meluap. Batang LMDB pada grafik di atas akan keluar dari grafik.

Gambar 32x32x3 piksel kami relatif kecil dibandingkan dengan rata-rata gambar yang mungkin Anda gunakan, dan memungkinkan kinerja LMDB optimal.

Meskipun kami tidak akan menjelajahinya di sini secara eksperimental, menurut pengalaman saya sendiri dengan gambar berukuran 256x256x3 atau 512x512x3 piksel, HDF5 biasanya sedikit lebih efisien dalam hal penggunaan disk daripada LMDB. Ini adalah transisi yang baik ke bagian akhir, diskusi kualitatif tentang perbedaan antara metode-metode tersebut.

## Diskusi

Ada fitur pembeda lainnya dari LMDB dan HDF5 yang perlu diketahui, dan penting juga untuk membahas secara singkat beberapa kritik terhadap kedua metode tersebut. Beberapa tautan disertakan bersama diskusi jika Anda ingin mempelajari lebih lanjut.

### Akses Paralel

Perbandingan utama yang tidak kami uji dalam eksperimen di atas adalah pembacaan dan penulisan [secara bersamaan .](https://realpython.com/python-concurrency/) **Seringkali, dengan kumpulan data sebesar itu, Anda mungkin ingin mempercepat operasi Anda melalui paralelisasi.**

Dalam sebagian besar kasus, Anda tidak akan tertarik membaca bagian dari gambar yang sama secara bersamaan, namun Anda _ingin_ membaca beberapa gambar sekaligus. Dengan definisi konkurensi ini, penyimpanan ke disk sebagai `.png`file sebenarnya memungkinkan konkurensi lengkap. Tidak ada yang menghalangi Anda membaca beberapa gambar sekaligus dari thread berbeda, atau menulis banyak file sekaligus, asalkan nama gambarnya berbeda.

Bagaimana dengan LMDB? Mungkin terdapat beberapa pembaca di lingkungan LMDB sekaligus, namun hanya satu penulis, dan penulis tidak memblokir pembaca. Anda dapat membaca lebih lanjut tentang hal itu di [situs web teknologi LMDB](http://www.lmdb.tech/doc/) .

Beberapa aplikasi dapat mengakses database LMDB yang sama secara bersamaan, dan beberapa thread dari proses yang sama juga dapat mengakses LMDB secara bersamaan untuk dibaca. Hal ini memungkinkan waktu baca yang lebih cepat: jika Anda membagi seluruh CIFAR menjadi sepuluh set, maka Anda dapat menyiapkan sepuluh proses untuk setiap pembacaan dalam satu set, dan ini akan membagi waktu pemuatan menjadi sepuluh.

HDF5 juga menawarkan I/O paralel, memungkinkan pembacaan dan penulisan secara bersamaan. Namun, dalam implementasinya, kunci tulis ditahan, dan akses dilakukan secara berurutan, kecuali Anda memiliki sistem file paralel.

Ada dua opsi utama jika Anda mengerjakan sistem seperti itu, yang dibahas lebih mendalam dalam [artikel ini oleh Grup HDF tentang IO paralel](https://www.hdfgroup.org/2015/04/parallel-io-why-how-and-where-to-hdf5/) . Ini bisa menjadi sangat rumit, dan opsi paling sederhana adalah membagi kumpulan data Anda menjadi beberapa file HDF5 secara cerdas, sehingga setiap proses dapat menangani satu `.h5`file secara terpisah.

### Dokumentasi

Jika Anda Google `lmdb`, setidaknya di Inggris, hasil pencarian ketiga adalah IMDb, Internet Movie Database. Bukan itu yang Anda cari!

Sebenarnya, ada satu sumber dokumentasi utama untuk pengikatan Python pada LMDB, yang dihosting di [Read the Docs LMDB](https://lmdb.readthedocs.io/en/release/#) . Meskipun paket Python bahkan belum mencapai versi > 0.94, paket ini _cukup_ banyak digunakan dan dianggap stabil.

Sedangkan untuk teknologi LMDB sendiri, terdapat dokumentasi yang lebih detail [di situs web teknologi LMDB](http://www.lmdb.tech/doc/index.html) , yang mungkin terasa seperti belajar kalkulus di kelas dua, kecuali Anda memulai dari halaman [Memulainya](http://www.lmdb.tech/doc/starting.html) .

Untuk HDF5, terdapat dokumentasi yang sangat jelas di situs [dokumen h5py](http://docs.h5py.org/en/stable/) , serta [postingan blog bermanfaat oleh Christopher Lovell](https://www.christopherlovell.co.uk/blog/2016/04/27/h5py-intro.html) , yang merupakan ikhtisar luar biasa tentang cara menggunakan `h5py`paket tersebut. Buku O'Reilly, [Python dan HDF5](https://realpython.com/asins/1449367836/) juga merupakan cara yang baik untuk memulai.

Meskipun tidak terdokumentasi seperti yang mungkin disukai oleh pemula, baik LMDB maupun HDF5 memiliki komunitas pengguna yang besar, sehingga pencarian Google yang lebih dalam biasanya memberikan hasil yang bermanfaat.

### Pandangan yang Lebih Kritis terhadap Implementasi

Tidak ada utopia dalam sistem penyimpanan, dan baik LMDB maupun HDF5 memiliki kelemahan masing-masing.

Hal penting yang perlu dipahami tentang LMDB adalah bahwa data baru ditulis **tanpa menimpa atau memindahkan data yang sudah ada.** Ini adalah keputusan desain yang memungkinkan pembacaan sangat cepat yang Anda saksikan dalam eksperimen kami, dan juga menjamin integritas dan keandalan data tanpa perlu lagi menyimpan log transaksi.

Namun ingat, Anda perlu menentukan `map_size`parameter alokasi memori _sebelum_ menulis ke database baru? Di sinilah LMDB bisa merepotkan. Misalkan Anda telah membuat database LMDB, dan semuanya baik-baik saja. Anda telah menunggu dengan sabar hingga kumpulan data Anda yang sangat besar dimasukkan ke dalam LMDB.

Kemudian, di kemudian hari, Anda ingat bahwa Anda perlu menambahkan data baru. Bahkan dengan buffer yang Anda tentukan pada `map_size`, Anda mungkin akan melihat `lmdb.MapFullError`kesalahan dengan mudah. Kecuali Anda ingin menulis ulang seluruh database Anda, dengan update `map_size`, Anda harus menyimpan data baru tersebut dalam file LMDB terpisah. Meskipun satu transaksi dapat mencakup beberapa file LMDB, memiliki banyak file masih menyusahkan.

Selain itu, beberapa sistem memiliki batasan mengenai jumlah memori yang dapat diklaim sekaligus. Berdasarkan pengalaman saya sendiri, bekerja dengan sistem komputasi kinerja tinggi (HPC), hal ini terbukti sangat membuat frustrasi, dan sering kali membuat saya lebih memilih HDF5 daripada LMDB.

Dengan LMDB dan HDF5, hanya item yang diminta yang dibaca ke dalam memori sekaligus. Dengan LMDB, pasangan unit kunci dibaca ke dalam memori satu per satu, sedangkan dengan HDF5, `dataset`objek dapat diakses seperti array Python, dengan pengindeksan `dataset[i]`, rentang, `dataset[i:j]`dan splicing lainnya `dataset[i:j:interval]`.

Karena cara sistem dioptimalkan, dan bergantung pada sistem operasi Anda, urutan akses item dapat memengaruhi kinerja.

Menurut pengalaman saya, secara umum benar bahwa untuk LMDB, Anda mungkin mendapatkan kinerja yang lebih baik ketika mengakses item secara berurutan berdasarkan kunci (pasangan nilai-kunci disimpan dalam memori yang diurutkan secara alfanumerik berdasarkan kunci), dan untuk HDF5, mengakses rentang yang besar akan berkinerja lebih baik daripada membaca setiap elemen kumpulan data satu per satu menggunakan yang berikut:

Jika Anda mempertimbangkan pilihan format penyimpanan file untuk menulis perangkat lunak Anda, maka akan lalai untuk tidak menyebutkan [Menjauh dari HDF5](https://cyrille.rossant.net/moving-away-hdf5/) oleh Cyrille Rossant tentang jebakan HDF5, dan tanggapan Konrad Hinsen [Tentang HDF5 dan masa depan manajemen data](http://blog.khinsen.net/posts/2016/01/07/on-hdf5-and-the-future-of-data-management/) , yang mana menunjukkan bagaimana beberapa kendala dapat dihindari dalam kasus penggunaannya dengan banyak kumpulan data yang lebih kecil daripada beberapa kumpulan data yang sangat besar. Perhatikan bahwa kumpulan data yang relatif lebih kecil masih berukuran beberapa GB.

### Integrasi Dengan Perpustakaan Lain

Jika Anda berurusan dengan kumpulan data yang sangat besar, kemungkinan besar Anda akan melakukan sesuatu yang signifikan dengan kumpulan data tersebut. Sebaiknya pertimbangkan perpustakaan pembelajaran mendalam dan jenis integrasi apa yang ada dengan LMDB dan HDF5.

Pertama-tama, semua perpustakaan mendukung pembacaan gambar dari disk sebagai `.png`file, selama Anda mengonversinya menjadi array NumPy dengan format yang diharapkan. Hal ini berlaku untuk semua metode, dan kita telah melihat di atas bahwa membaca gambar sebagai array relatif mudah.

**Berikut adalah beberapa perpustakaan pembelajaran mendalam paling populer serta integrasi LMDB dan HDF5-nya:**

-   [**Caffe**](https://caffe.berkeleyvision.org/) memiliki integrasi LMDB yang stabil dan didukung dengan baik, serta menangani langkah membaca secara transparan. Lapisan LMDB juga dapat dengan mudah diganti dengan database HDF5.
    
-   [**Keras**](https://www.tensorflow.org/tutorials/keras/save_and_restore_models) menggunakan format HDF5 untuk menyimpan dan memulihkan model. Artinya, TensorFlow juga bisa melakukannya.
    
-   [**TensorFlow**](https://www.tensorflow.org/api_docs/python/tf/contrib/data/LMDBDataset) memiliki kelas bawaan`LMDBDataset`yang menyediakan antarmuka untuk membaca data masukan dari file LMDB dan dapat menghasilkan iterator dan tensor dalam batch. [TensorFlow](https://realpython.com/pytorch-vs-tensorflow/) tidak _memiliki_ kelas bawaan untuk HDF5, tetapi dapat ditulis kelas yang mewarisi`Dataset`kelas tersebut. Saya pribadi menggunakan kelas khusus yang dirancang untuk akses baca optimal berdasarkan cara saya menyusun file HDF5 saya.
    
-   [**Theano**](http://deeplearning.net/software/theano/) pada dasarnya tidak mendukung format file atau database tertentu, tetapi seperti yang disebutkan sebelumnya, Theano dapat menggunakan apa pun asalkan dibaca sebagai array berdimensi-N.
    

Meskipun jauh dari komprehensif, semoga ini memberi Anda gambaran tentang integrasi LMDB/HDF5 dengan beberapa perpustakaan pembelajaran mendalam utama.

## Beberapa Wawasan Pribadi tentang Menyimpan Gambar dengan Python

Dalam pekerjaan saya sehari-hari menganalisis gambar medis berukuran terabyte, saya menggunakan LMDB dan HDF5, dan telah mempelajari bahwa, dengan metode penyimpanan apa pun, **pemikiran ke depan sangatlah penting** .

Seringkali, model perlu dilatih menggunakan k-fold cross validation, yang melibatkan pemisahan seluruh kumpulan data menjadi k-set (k biasanya berjumlah 10), dan k model dilatih, masing-masing dengan k-set berbeda yang digunakan sebagai set pengujian. Hal ini memastikan bahwa model tidak melakukan overfitting pada kumpulan data, atau, dengan kata lain, tidak dapat membuat prediksi yang baik pada data yang tidak terlihat.

Cara standar untuk membuat k-set adalah dengan menempatkan representasi yang sama dari setiap jenis data yang direpresentasikan dalam dataset di setiap k-set. Oleh karena itu, menyimpan setiap k-set ke dalam kumpulan data HDF5 terpisah akan memaksimalkan efisiensi. Terkadang, satu k-set tidak dapat dimuat ke dalam memori sekaligus, sehingga bahkan pengurutan data dalam suatu dataset memerlukan pemikiran terlebih dahulu.

Dengan LMDB, saya juga berhati-hati dalam membuat rencana ke depan sebelum membuat database. Ada beberapa pertanyaan bagus yang patut ditanyakan sebelum Anda menyimpan gambar:

-   Bagaimana cara menyimpan gambar sedemikian rupa sehingga sebagian besar pembacaannya berurutan?
-   Apa kunci yang bagus?
-   Bagaimana saya bisa menghitung dengan baik `map_size`, mengantisipasi potensi perubahan di masa depan dalam kumpulan data?
-   Seberapa besar suatu transaksi, dan bagaimana seharusnya transaksi dibagi lagi?

Apa pun metode penyimpanannya, saat Anda berurusan dengan kumpulan data gambar berukuran besar, sedikit perencanaan akan sangat membantu.

## Kesimpulan

You’ve made it to the end! You’ve now had a bird’s eye view of a large topic.

In this article, you’ve been introduced to three ways of storing and accessing lots of images in Python, and perhaps had a chance to play with some of them. All the code for this article is in a [Jupyter notebook here](https://github.com/realpython/materials/blob/storing-images/storing-images/storing_images.ipynb) or [Python script here](https://github.com/realpython/materials/blob/storing-images/storing-images/storing_images.py). Run at your own risk, as a few GB of your disk space will be overtaken by little square images of cars, boats, and so on.

You’ve seen evidence of how various storage methods can drastically affect read and write time, as well as a few pros and cons of the three methods considered in this article. While storing images as `.png` files may be the most intuitive, there are large performance benefits to considering methods such as HDF5 or LMDB.

Feel free to discuss in the comment section the excellent storage methods not covered in this article, such as [LevelDB](https://github.com/google/leveldb), [Feather](https://github.com/wesm/feather), [TileDB](https://tiledb.io/), [Badger](https://blog.dgraph.io/post/badger/), [BoltDB](https://godoc.org/github.com/boltdb/bolt), or anything else. **There is no perfect storage method, and the best method depends on your specific dataset and use cases.**

## Further Reading

Here are some references related to the three methods covered in this article:

-   [Python binding for LMDB](https://lmdb.readthedocs.io/en/release/#)
-   [LMDB documentation: Getting Started](http://www.lmdb.tech/doc/starting.html)
-   [Python binding for HDF5 (h5py)](https://www.h5py.org/)
-   [The HDF5 Group](https://www.hdfgroup.org/solutions/hdf5/)
-   [“Python and HDF5” from O’Reilly](https://realpython.com/asins/1449367836/)
-   [Pillow](https://pillow.readthedocs.io/en/stable/)

You may also appreciate [“An analysis of image storage systems for scalable training of deep neural networks”](https://www.osti.gov/biblio/1335300) by Lim, Young, and Patton. That paper covers experiments similar to the ones in this article, but on a much larger scale, considering cold and warm cache as well as other factors.