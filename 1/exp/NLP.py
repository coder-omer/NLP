#%% NLP
# 8 session - 1 lab - 1 PROJECT(ML,DL,NLP karışımı)

#%% NLP-1
# NLP : Natural Language Processing
# CNN ve NLP özellikle uzmanlaşılacak alanlardır
# Çoğunlukla classification üzerinde yoğunlaşacağız. Diğer uygulamaları hazır paketler üzerinden göstereceğiz
# Junior seviyede pozisyonlara(NLP için) hazır olacaksınız

# SofiaH: NLP: ML, DL ve AI alanlarının hepsinden faydalanır. Text'lerin makinenin anlayabileceği forma dönüştürülmesidir
# .. ML modellerinde vectorization işlemleri ile, 
# .. DL modellerinde TF-IDF vectorizer işlemleri ile text'ler numeric hale getirilerek modellere verilir

# NLP nedir?
# Insan dilinin bazı algoritma, model vs tarafından anlaşılıp kullanıldığı yöntem.

# NLP neden bu kadar patladı?
# ML, DL de kullandığımız datalar nasıl datalardı. Tabular(Satır ve sütunlardan oluşan / Yapılandırılmış) datalardı(CNN hariç)
# Ancak her türlü datayı satır ve sütunlara dönüştürmemiz kolay değil
# NLP ile datamızı yapılandırılmış dataya dönüştürmeden yazı üzerinde bizim istediğimiz keyword ler üzerine yoğunlaşıp classification yapabiliyoruz

# NLP kullanım alanları
# NOT: (johnsnowlabs firmasında) bir araştırmada doktorlar %60-65 civarı doğru teşhis yaparken makineler %80 üzerinde doğru tahmin yapmışlar
    # Diagnosing(Healthcare)
        # SofiaH: Hastaların tedavi süresince yapılan işlemler modele verilir. Model text verilerine veya labdan gelen sayısal verilere
        # .. dayanarak hastalık analizi yapar.
    # Sentiment Analysis
        # SofiaH: Firmalar, müşterilerin yaptıkları yorumlar üzerinden kendini geliştirmeye çalışıyor. Hükümetler de kanun çıkarırken yapılan 
        # .. yorumlara göre hareket ediyorlar
    # ChatBot
    # Machine Translation
        # SofiaH: Google translate bunu kullanıyor
    # Digital Assistant(Speech Recognation)
        # SofiaH: Telefon şirketlerini aradığımızda karşımıza çıkan digital aşistan. Konuşmayı "speech recognition" programı
        # .. ile text'e çevirir ve sonra arka planda NLP yi kullanır(Amazon Echo, Google Home)

# Digital assistanlara örnek
# Verilen sesli komutları text e çevirip bunlar anlamlandırılıp ona göre makine bir anlam çıkarıyor
# Ancak çok başarılı değiller. Mesela tel paketinizi değiştirmek istediğinizde sürekli tekrar eden cevaplar veriyor

# Google ın geliştirdiği digital assistant-3 yıllık teknoloji
# Google ın eğittiği model kuaförü arayacak ve bir randevu almaya çalışacak alttaki linkte
# https://www.youtube.com/watch?v=D5VN56jQMWM
# Bunları yapabilmek için çok büyük datalara sahip olmak lazım(Google, Microsoft vs gibi firmalar)(Gmail de atılan mailler, çeviriler vs vs)
# Örneğin; Google translate de günlük 13 milyar çeviri yapılıyor. Ortalama 2 milyarı yeni kullanılan cümleler
# Bizim eğittiğimiz modeller bazen yetersiz kalıyor, google gibi firmaların eğittiği hazır modelleri kullanacağız bu durumlarda

# NLP teori kısmı ve terimler
# NOT: Piyasa da en çok kullanılan sentiment analysis ve classification a bakacağız burada
# NOT: Diğer uygulamaları hazır modeller üzerinden anlatacağız
# Corpora  : Bir çok corpus un bir arada olduğu alan. Buna NLP özelinde corpora denir. Database diyebiliriz
# Corpus   : Data seti. 
# Document : Corpustaki her bir satır. Bir kelimeden, bir cümleden, bir paragraftan, bir kaç sayfadan, bir kitaptan oluşabilir

"""
*********************************************************
# SofiaH
1. Tokenization
2. Remove punctuation and numbers
3. Remove stopwords
4. Data Normalization (Lemmatization-Stemming)
5. Count Vectorization
6. TF-IDF
**********************************************************
"""
##### Tokenization
# Data cleaning in 1. aşaması Tokenization dır
# SofiaH: 
# Elimizdeki mevcut text in makinalar tarafından daha kolay anlaşılabilmesi için daha küçük parçalara bölünmöesidir
# 2 farklı tokenization işlemi var. 1. sentence 2.word
# Çoğunlukla word tokenization kullanacağız. Sentence tokenization ın piyasa da çok kullanımı yok
# Slayttaki cümlede cümleyi ayrı kelimelere(tokenlere) ayırmışız.(Tokenleştirme yapılmış)
# Bunları sayısal hala dönüştürüp modele vereceğiz ve model bunlardan anlam çıkaracak
# Cümle olarak verirsek modele, modelin bunu anlamlandırması daha zor olacaktır. Bunun yerine belli başlı kelimeler
# .. kullanayım bu kelimelerle cümle oluşturma ve makina için anlaşılması daha kolay olacaktır.
# SofiaH: Datadaki kelimelerin hepsi modele saydırılır. Saydırma işlemi yapılmazsa CountVectorizer, TF-IDF Vectors işlemleri de yapılamaz

##### Lower
# Cümledeki kelimeleri numaralandırabilirim --> Örneğin: This is a sample
# This   --> 1 numara # is     --> 2 numara # a      --> 3 numara # sample --> 4 numara
# Tokenlerimiz numaralandırıp bu tokenler kendi arasında anlamsal ilişki kurabilir. Cümleler halinde verirsek model daha zor öğrenir
# Data Cleaning in 2. aşaması: Tokenlerimizi küçük harflere dönüştürmezsem model hepsine farklı token muamelesi
# ..  yapar(THIS, This, this --> Bu 3 ünü farklı 3 token olarak algılar)
# .. bu da modelimizin öğrenememesine(ya da yanlış öğrenmesine) yol açar.
# Bunları ML ve DL modellerinde kesinlikle yapmalıyız. Bundan sonraki aşamaları DL de ister yapın ister yapmayın

#### Remove Puncs and numbers
# Cleaning in 3. aşaması: Noktalama işaretlerin temizlenmesi
# Keyword leri aradığı için model bunları gürültü olarak görüyor
# Zaten 10000-15000 tane feature ımız olacak. Bunları da eklemeye gerek yok
# NOT: SofiaH: Classification ve sentimental analizlerde sayısal verilerde temizlenir

#### Remove stopwords
# Cleaning in 4. aşaması: stopwords leri çıkarma
# Stopwords: Datamızın anlamına extra bir derinlik katmayan kelimeler: bağlaçlar, sıklık zarfları, soru kelimeleri vs vs
# Örneğin: Onlar beğendiler, Onlar beğenmediler --> Burada 1 olumlu 1 olumsuz cümle olmasına karşın
# .. "onlar" kelimesi olumlu ya da olumsuz bir anlam vermiyor o yüzden bu kelimeyi boş yere almıyoruz(feature olarak)
# nltk kütüphanesini kullanacağız(179 tane bizim işimizi görecek). Bunları datanızda tutsanızda classification ve sentiment analizini güzel yapar
# .. ama hesaplama maliyetiniz olur.

##### Normalization(Lemma)
# Cleaning 5. aşama: Tokenlerin köklerine inme --> diğer bir ad ile "Normalization"
# ML de bunu yapacağız. DL de tokenlerin asıl halleriyle bulunması lazım
# .. Çünkü DL de anlamsal ilişki olduğu halleriyle kurulabilir
# "beğendin" mi "beğenmedin" mi bunu araştırıyoruz  . Beğendim, beğendin, beğenildi, beğenilmedi ..... vs vs
# Farklı varyasyonlarını iptal edip kelimenin kökenine inmek --> "beğenmek"
# 2 farklı yöntem vardır
# Lemmatization: Bu tokenin kökenine inersem anlam kaybı olur mu olmaz mı bakar(Arabacı, dondurmacı --> araba, dondurma --> anlam değişti)
# .. anlam kaybı olduğunu görürse ekleri atmaz "Lemmatization". Best practice olarak Lemmatization kullanılır ama
# .. Lemmatization kullanmak zorunlu değildir ama tavsiyem Lemmatization kullanmanızdır
# Stemming : Direk kökleri alır. Anlam değişiyormuş, değişmiyormuş bakmaz.
# .. Stemming bazen saçmalar. Johnson hoca: Mesela "koyun" kelimesine "koy" dediğini gördüm

# NOT: SofiaH: DL modellerinde data cleaning işlemlerini yapmaya gerek kalmaz bu işlemler genelde ML modellerinde yapılır
# NOT: SofiaH: Model, stopword'ler içindeki olumlu-olumsuz yardımcı fiillerle(should, couldn't) sentimental analysis yapacağı
# .. zaman ilgilenir. Bu tarz modellerde bu stopwordler atılMAMALIDIR.
# NOT: SofiaH: Stopword lerin kullanılan kütüphanelere göre sayıları : NLTK(179 English Stopwords),spaCy(179 English Stopwords),
# .. gensim(179 English Stopwords). Bunlar ML modelleri için kullanılan kütüphanelerdir, DL'de pek kullanılmazlar
# Biz genelde DL modellerini kullanacağız. DL modelleri MŞ modellerine göre daha iyi sonuç verirler fakat bazen istisnai durumlarla karşılaşacağız

# Örnek: Classification dan ziyade alttaki örneği sentiment analizi(olumlu-olumsuz) şeklinde değerlendirmek daha iyi
# Sample_text = "Oh man, this is pretty cool. We will do more such things."
    # Tokenization : ['oh','man',',','this', 'is','pretty','cool','.','we','will','do','more','such','things','.']
    # Removing punctation: ['oh','man','this','is','pretty','cool','we','will','do','more','such','things']
    # Removing stopwords : ['oh','man','pretty','cool','things']
    # stemming : ['oh','man','pretti','cool','thing']
    # lemmatization instead of stemming : ['oh','man','pretty','cool','thing']
    
# Model "Pretty" ve "cool" vs gördüğünde büyük ihtimalle bir sonuç döndürecek zaten. Olayın ana fikrine bakıyoruz yani
# .. Bu cümlenin tamamını hatırlamayacağız ama bu cümlenin olumlu mu olumsuz mu olacağını bileceğiz. Model de bu şekilde çalışıyor
# "Pretty" nin "güzel" anlamının yanı sıra "oldukça" anlamı olmasına rağmen "pretti" şeklinde almış stemming de
# .. yani anlamı sadece "güzel" olan kelime köküne inilmiş.

# vectorizer.get_geature_names() : datamızda geçen bütün unique tokenleri tespit edip(daha sonra featurelara dönüştüreceğiz bunları)

##### SAYISALA ÇEVİRME
# Son aşamadır. Data temizleme işlemi bittikten sonra ham datayı modelin anlayabileceği şekilde sayısal forma çevirmemiz gerekiyor
# 1. Count vectorizer(countvectorizer) 2.TF-ITF(tfidfvectorizer) 3.Word Embedding(Word2Vec & Glove)
# Word embedding en advanced algoritmadır
# ilk 2 si ML modellerinde tercih edilir

##### 1-Countvectorizer Yöntemi
# Corpusumuzdaki bütün yorumlarda geçen unique tokenleri elde ediyoruz önce sonra bunları herbirini stopwordlerden 
# .. temizlenmiş şekilde alfabetik sıraya göre birer feature yaparak yazar
# SofiaH: CountVectorizer, keyword'lere öncelik tanırken, document da ne kadar sıklıkta geçtiğine dikkat eder
# .. Cümle içinde hangi tokenden kaçar tane var bunu sayar ve sayısı fazla olana fazla ağırlık verir
# SofiaH: Yukarda örnekte CountVectorizer, "likes" ve "movies" e çok fazla ağırlık verecektir
# SofiaH: CountVectorizer document içindeki hangi tokenin önemli olduğunu tespit edebilir fakat corpus içinde
# .. ne kadar önemli olduğunu tespit edemez.
# SofiaH: Modele olumlu ve olumsuz anlam katan token sayısı eşit ise CountVectorizer yorumun olumlu veya olumsuz olduğunu anlayamayabilir
# Document-1: John likes to watch movies. Mary likes movies too
# Document-2 : Mary also likes to watch football games
# Corpusumuz: document-1 ve document-2 den oluşan kısım

"""
    also football games john likes mary movies to too watch
1 : 0       0       0    1     2     1   2     1   1   1
2 : 1       1       1    0     1     1   0     1   0   1
"""

##### 2.TF-IDF Yöntemi
# Count vektorizer a göre daha gelişmiş bir modeldir. Ama bazen countvectorizer ile daha yüksek skorlar alabiliriz
# Countvektorizer da bir tokenin cümle içinde geçme sıklığına göre numaralandırma yapıyorduk
# .. Bir token hem corpus içerisinde hem de o yorum özelinde ne kadar kullanılmış insight ını vermiyordu count vektorizer
# .. bu insight ı TF-IDF veriyor. Yani
# TF-IDF: Bir token hem corpus içerisinde hem de o yorum özelinde ne kadar kullanılmış insight ını bize vererek hesaplama yapar
# TF-IDF kullanacaksanız stopword leri silmeden devam edebilirsiniz ama yine de silerek devam etmeniz daha iyii
# .. çünkü TF-IDF ona göre sıklıkla geçen bu stopwordlerin katsayılarını küçük tutacak.
# .. Eğer bir tokenim bütün yorumlarda geçiyorsa o token üzerinden bir duygu analizi yapamazsanız anlamına geliyor

# TF : Bir tokenin yorum içerisindeki kullanım sıklığı( .. ile alakalı bir insight veriyor) 
# "Ahmet Tülini beğendi" --> "beğendi" için --> TF = 1/3 

# IDF: Bir tokenin corpus içerisindeki geçme sıklığı (... ile alakalı insight sağlıyor)
# Inverse kısmından önce Document frequency ye bakalım
# 100 yorumum olsun. 1. yorumumda , 2. yorumumda, 100. yorumumda "beğendi" kelimesi geçsin --> DF = 3 /100
# Inverse ü de ekleyelim. yani tersini --> 100/3  # NOT: Eksi değerden kurtulmak için Inverse işlemi yapıyoruz
# Sonra buna log ekleriz --> IDF = log(100/3)
# Peki neden log aldık(np.log(3/100) = -1.52 , np.log(100/3) = 1.52, --> np.log(1000000/3)=6.52)
# log bir nevi doğal scale yapıyor. Çalışma maliyetini azaltıp ağırlıklandırmayı ayarlıyor
# .. yani bazı featurelara fazla ağırlık verilip benim için önemli olan tokenimi modelimin kaçırmasını
# .. engellemek için log ile doğal bir scaleleme yapıyoruz
# 

# SONUÇ: Örneğin Bu 2 değeri çarpıyoruz : ÖrneğTF * IDF = 0,15 diyeceğiz

# Örnek hesaplama: yorum 100 kelimeden oluşsun "cow" kelimesi 3 kere geçsin
# 10 milyon dökümanda 1000 defa geçiyor olsun "cow" kelimesi
# Katsayı hesabı = 0.03 * 4 = 0.12
# TF-IDF data özelinde bu token önemli veya değil anlamında bir insight verir. (Yani örneğin:Olumlu-olumsuz şeklinde insight vermez)
# Yani TF-IDF özelinde katsayısı düşük olan tokenimiz de bizim için önemli bir keyword olabilir.

# Uygulamada yapacağımız aşamalar
    # Data cleaning yapacağız
    # Metni, sayısal forma dönüştüreceğiz
    # Modele veip sonuç alacağız

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 50)

#!pip install nltk # Not: anaconda kullananların install etmesine gerek yok

import nltk

# Nltk kütüphanesi bazı sürümlerde bu alttakilerin download edilmesini istiyor. 
# Yani import ettiğiniz halde hata alıyorsanız bunları download ediniz.
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

sample_text = "Oh man, this is pretty cool. We will do more such things. 2 ½ % ()" 
# sayısal değerleri ve özel karakterleri nasıl temizlediğini görmek için bazı şeyler ekledik cümleye

from nltk.tokenize import sent_tokenize, word_tokenize
sentence_token = sent_tokenize(sample_text.lower())  # sentence tokenlere ayır ve küçük harflere dönüştür
sentence_token 
# Sent_tokenize: sentence tokenizer. Bunu çok kullanmayacağımızı söylemiştik Noktayı dikkate alarak ayırma işlemi yapıyor.
# SofiaH: sent_tokenize --> Sentence' lari bozmadan tokenize islemini yapar ve lower ile bütün harfleri kücük harfe cevirir :

word_token = word_tokenize(sample_text.lower())  # word tokenlere ayır ve küçük harflere dönüştür
word_token
# word_tokenize: Boşlukları dikkate alarak ayırma işlemi yapıyor(NOT: Sadece string ifadelerde boşluk arar)
# SofiaH  word_tokenize --> Kelime kelime ayrim yaparak tokenize islemini yapar ve lower ile bütün harfleri kücük harfe cevirir.
# .. Noktalama isaretleri de birer token olarak kabul edilir

##### Removing Punctuation and Numbers
tokens_without_punc = [w for w in word_token if w.isalpha()] # .isalnum() for number and object
tokens_without_punc
# isalpha = Check if all the characters in the text are letters
# isalpha() : Çektiğim ifadenin string ise TRUE deyip(... list comprehension la kelimeleri alıyoruz)
# Bu aşamada Noktalama işaretleri, sayısal değerler ve özel karakterleri temizliyoruz 
# String le birlikte sayısal değerlerinde kalmasını istiyorsanız .isalnum() kullanılabilir
# SofiaH: Tokenization isleminden sonraki ikinci asama olarak noktalama isaterlerinden ve sayilardan kurtulmamiz gerekiyor.
# SofiaH: ML ile hazirlanan modellerde classification veya sentimental analysis yapilabiliyor. Bu analizlerde de sayilar ve 
# .. noktalama isaretlerini temizlemek gerekir.
# SofiaH: isalpha --> Tokenin object (str) ifade olup olmadigina bakar; object ise gecirir fakat noktalama isareti veya sayisal deger ise
# .. gecirmez. (Sayilarin da kalmasi isteniyor ise isalpha yerine .isalnum() yazilabilir.

##### Removing Stopwords
# SofiaH: Cleaning islemini iki farkli sekilde yapacagiz : Classification islemi icin, sentimental analysis icin(olumlu veya olumlu sonuc
# ..  bizim icin onemli ise).
from nltk.corpus import stopwords
stop_words = stopwords.words("english") # SofiaH: stop_words isimli degiskenin icine hangi dilin stopword' lerini kullanacaksak onu tanimladik
stop_words

tokens_without_punc # SofiaH: Corpus' u (data) word tokenler haline getirmistik. Bu datadan stopword' leri cikaracagiz 

token_without_sw = [t for t in tokens_without_punc if t not in stop_words] # if you don't make a sentiment analysis , 
                                                                           # you can remove negative auxiliary verb
token_without_sw # Output : ['oh', 'man', 'pretty', 'cool', 'things']
# stop_words lerde gez eğer cümlem(tokens_without_punc) içinde stop words(stop_words değişkeni) içinde değilse liste 
# .. içerisinde tut, değilse ignore et
# NOT: Eğer sentiment analizi yapmayacaksanız negatif yardımcı fiilleri çıkarabilirsiniz
# .. Eğer sentiment analizi yapacaksak bunları datada tutmamız lazım çünkü(no, not vs..) örneğin;
    # It is a problem     (Olumsuz sonuç gelecek)
    # It is not a problem (Olumlu sonuç gelecek)

##### Data Normalization-Lemmatization
# SofiaH: NLP' de data normalization islemi Lemmatization veya Stemming ile yapilir. Lemmatization sözlükteki anlami korudugu icin daha cok 
# .. tercih edilen bir yöntemdir.
from nltk.stem import WordNetLemmatizer
WordNetLemmatizer().lemmatize("drives") 
# Örnek: "driving" yazsaydık bunun farklı anlamı olduğunu bildiği için kökenine inmeyip olduğu gibi sonuç verirdi
# Örnek: "drove" yazarsak bunun "sürü" anlamı olduğunu da bildiği için olduğu gibi gelirdi
# Örnek: Children --> child olacak 

lem = [WordNetLemmatizer().lemmatize(t) for t in token_without_sw]
lem # Output : ['oh', 'man', 'pretty', 'cool', 'thing']
# SofiaH: Stopword' lerden temizlenmis corpus' un icindeki her bir tokeni list comprehension yöntemi ile köklerine indirmis olduk

##### Data Normalization-Stemming
from nltk.stem import PorterStemmer
PorterStemmer().stem("children") # Output : 'children'

stem = [PorterStemmer().stem(t) for t in token_without_sw]
stem  # Output: ['oh', 'man', 'pretti', 'cool', 'thing']
# Her bir tokenin köklerine iniyoruz
# Not: Çoğul ekinin de anlamı olmadığı için atmış          # Output: 'oh man pretty cool thing'

##### Joining
" ".join(lem) # lem içerisindeki tokenleri al aralarında birer boşluk bırak ve birleştir
# SofiaH: Liste icindeki bütun tokenleri join ile birlestirdik
# Her birini tek tek yapmak yerine bunu bir fonksiyona bağlayalım

##### Cleaning Function - for classification (NOT for sentiment analysis)
def cleaning(data):
    #1. Tokenize and lower
    text_tokens = word_tokenize(data.lower())    
    #2. Remove Puncs and numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]   
    #3. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]   
    #4. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]    
    #joining
    return " ".join(text_cleaned)
# Classification yaparken bu fonksiyonu kullanabiliriz.
# SofiaH: Eger bir classification islemi yapacaksak, herhangi bir duygu analizi yapmayacaksan asagidaki fonksiyonu kullanabiliriz 

pd.Series(sample_text).apply(cleaning) # Example: df["text"].apply(cleaning) # Output: 0    oh man pretty cool thing
# apply ı serilere uygulayabildiğimiz için seriye dönüştürdük.

#%% NLP-2
#### Cleaning Function - for sentiment analysis
# SofiaH: Eger sentimental bir analiz yapacaksak olumlu-olumsuz yardimci fiilerin text' in icinde kalmasi önemlidir.
sample_text= "Oh man, this is pretty cool. We will do more such things. don't aren't are not. no problem"

# SofiaH: Asagida "Text' in icinde bir (') var ise bunun yerine hicbir sey atama diyerek bunu bir degiskene("s") atadik.
# SofiaH: Bu degiskeni word_tokenize icine vererek tokenlerine ayirdik. Ayraci kaldirdigimiz icin arent kelimesi stopword icindeki 
# .. aren't kelimesi ile eslesmeyecek ve stopword isleminden sonra da bu kelimeler corpus icinde olmaya devam edecek :
s = sample_text.replace("'",'')
word = word_tokenize(s)
word 

# SofiaH: Bazen aren't yerine are not da kullanilabilir. Bu ayri yazimlarin da stopword asamasinda temizlenmesini engellememiz gerekir. 
# .. Bunun icin asagida bir fonksiyon tanimladik.
# SofiaH: Ilk olarak, yardimci fiillerdeki ayraclari kaldirdik.
# SofiaH: Ikinci olarak, bunlari word_tokenlerine ayirdik ve kücük harflere dönüstürdük.
# SofiaH: Ücüncü olarak, numaralardan ve noktalama isaretlerinden temizledik.
# SofiaH: Dördüncü olarak, stopword asamasinda bir for döngüsü kurarak 'not' ve 'no' sözcüklerini stopword' ler arasindan kaldirdik ki bu kelimeler
# .. stopword isleminden sonra da datamizda kalmaya devam etsin. Daha sonra list comprehension ile stopword' lerden temizleme islemini yaptik.
# SofiaH: Besinci olarak lemmatization islemi ile tokenlerin köklerine indik.
for i in ["not", "no"]:
    stop_words.remove(i)
def cleaning_fsa(data):
    #1. removing upper brackets to keep negative auxiliary verbs in text
    text = data.replace("'",'')
    #2. Tokenize
    text_tokens = word_tokenize(text.lower()) 
    #3. Remove numbers
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]
    #4. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]
    #5. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]
    
    #joining
    return " ".join(text_cleaned)

stop_words

# SofiaH: sample_text' i seri haline getirmeden apply islemi uygulayamiyoruz. Text' i seri haline getirdikten sonra yukarida olusturdugumuz 
# .. fonksiyonu apply ile ekledik. Böylece tek satirda cleaning islemi tamamlanmis oldu :
pd.Series(sample_text).apply(cleaning_fsa)

##### CountVectorization and TF-IDF Vectorization
df = pd.read_csv("airline_tweets.csv")
df.head()
# SofiaH: Bir havayoluyla ilgili atilan tweet yorumlarindan olusan bir corpus var. Bu corpus üzerinden CountVectorization ve TF-IDF Vectorization 
# .. islemlerinin mantiginin nasil isledigini görecegiz.

df = df[['airline_sentiment','text']]
df # Sofia H: NLP datalarinin hepsi, text ve label olarak 2 feature' a düsürülür. Bu yüzden corpus' tan sadece bu iki sütunu aldik 

df = df.iloc[:8, :]
df # SofiaH: df' teki ilk 8 satiri ve tüm feature' lari aldik (Daha anlasilir olmasi icin) :

df2 = df.copy() # SofiaH:  df' in bir kopyesini df2' ye atadik. (Bu sekilde yapmazsak hata veriyor) :

df2["text"] = df2["text"].apply(cleaning_fsa) 
df2
# SofiaH: df2' deki text'e apply ile yukarida olusturdugumuz cleaning_fsa fonksiyonunu uyguladik 
# .. ve boylece df icin celaning islemini uygulamis olduk (Duygu analizi icin olusturdugumuz fonksiyon).
# NOT: SofiaH: !! Model kurarken cumle icindeki grammer yapisindan dolayi sira onemli fakat cumleler arasi sira onemli degil. _!!

##### CountVectorization
# SofiaH: CountVectorizer ile text' leri sayisal hale dönüstürme islemi yapacagiz.
X = df2["text"]                   # Yorumlar
y = df2["airline_sentiment"]      # Target label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, stratify = y, random_state = 42) 
# Datamizda toplam 8 cumle var, bunlari train ve test olarak yari yariya ayirdik 

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
# SofiaH: ML ve DL' deki scale isleminde yaptigimiz islemleri yapiyoruz; X_train'e fit_transform, X_test'e sadece transform islemi. (Data leakage'i 
# .. engellemek icin)
# SofiaH: X_train'e fit uygulandiginda, X_train icindeki unique bütün tokenler tespit edilir; transform ile ise döküman icindeki her token sayilir.
# SofiaH: X_test'e transform islemi uygulandiginda, dökümandaki sayma islemlerini X_train' e göre yapar. Örnegin X_test' te 'car' kelimesi var 
# .. fakat X_train' de bu kelime yoksa bu kelimeyi es gecer. Cünkü egittigimiz döküman icinde car kelimesi gecmiyor.
# SofiaH: Yani transform islemi, X_train' deki unique tokenlere göre yapilir.
# SofiaH: Bu yüzden X_train' i olabildigince buyuk tutmak gerekir ki tum tokenleri içersin.

vectorizer.get_feature_names()   # SofiaH: vectorizer' da egitilen unique token isimleri, feature isimleri olarak atandi :
# SofiaH: vectorizer.get_feature_names_out() --> Yeni versiyonlarda boyle.

X_train_count.toarray() # X_train' i array' e cevirdik ve her döküman icindeki tokenlerin teker teker sayildigini görmüs olduk

df_count = pd.DataFrame(X_train_count.toarray(), columns = vectorizer.get_feature_names())
df_count  # SofiaH: Array halindeki X_train datasini DafaFrame' e dönüstürdük. Columns isimleri olarak da get_feature_names' leri verdik. 
# .. Her dökümanda her tokenin kac kere gectigini görüyoruz ::

X_train # SofiaH: Yukaridaki DataFrame ile kiyaslamak icin asagida gercek X_train datasini yazdirdik, kelimelerin gercekte kacar kere gectigini 
# .. kiyaslamis olduk

X_train[6]   # SofiaH: 0. indexteki cumle.

vectorizer.vocabulary_ # SofiaH: vectorizer.vocabulary_ --> X_train' de gecen token sayilari.

##### TF-IDF
# sklearn TD-IDF https://towardsdatascience.com/how-sklearns-tf-idf-is-different-from-the-standard-tf-idf-275fa582e73d
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer()
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
"""
# SofiaH:
# TfidfVectorizer()' i bir degiskene atadik. Yine fit ve transform islemlerini yaptik.
# Yukaridaki CountVectorizer ile yaptigimiz fit_transform islemi ile buradaki farkli. 
# fit dedigimizde;
   # 1. Her satirda gecen unique tokenleri tespit eder (kac defa gectigine bakmaz), 
   # 2. Her cumlede gecen tokenleri sayar, bunlar her satirda var mi yok mu buna bakar (saymaz). 
# transform dedigimizde; 
   # 1. Unique tokenler kac satirda geciyor,
   # 2. Ilgili token her cumlede kac defa geciyor?
# ..  islemlerini yapar. Bu islemlerden sonra buldugu sayilari TF-IDF formullerinde yerlerine koyarak tokenleri sayisal veriler haline dönüstürür.

# NOT : X_train icin yaptigi transform islemini X_test icin de yapar fakat X_test' te gecen bir token X_train' de gecmiyorsa o tokeni görmezden
# .. gelir. Boyle bir durumda IDF hesabi yapilirken deger yerine konuldugunda log(0/0)=sonsuz olacaktir. Bunun onune gecmek icin IDF, 
# .. degerlere 1 ekler. log((0+1)/(0+1)). Bu sekilde degerlerin NaN cikmasinin önüne gecer.
"""

tf_idf_vectorizer.get_feature_names()      # SofiaH: tf_idf_vectorizer.get_feature_names_out()

X_train_tf_idf.toarray()

df_tfidf = pd.DataFrame(X_train_tf_idf.toarray(), columns = tf_idf_vectorizer.get_feature_names())
df_tfidf # SofiaH: DataFrame'e cevirip unique token isimlerini verdik ve yeni feature' lar olustu :

X_train[6]  # Output: 'virginamerica yes nearly every time fly vx ear worm go away'
# SofiaH: X_train' in 6. degerini yine kiyaslama icin aldik (Sifirinci index' teki deger). Yukarda ilk indexe bakacak olursak virginamerica 
# .. tokeni neredeyse her satirda gectigi icin stopword gibi kabul edilmis ve agirligi azaltilmis. Bir token her satirda gecerse önemsizlesir. 
# .. Bu tokenlere classification yapilamaz. (CountVectorizer, virginamerica kelimesini onemsizlestirmemisti fakat TF-IDF onemsizlestirdi)

df_tfidf.loc[2].sort_values(ascending=False) # SofiaH: Datanin 2. indexinde var olan kelimelerden en dusuk agirliga sahip olan tokeni, virginamerica

pd.DataFrame(X_test_tf_idf.toarray(), columns = tf_idf_vectorizer.get_feature_names()) 
# SofiaH: X_testi DataFrame donusturduk (X_train' deki feature' lara gore)

X_test    
X_test[3] 
# SofiaH: X_test' in ilk cumlesi 3. cumle. Bu cumlede gecen aggressive kelimesinin X_test feature' lari arasinda olmadigini goruyoruz.
# .. Demek ki bu kelime X_train' de yokmus ve gözardi edilmis. Bu durumda tahmin asamasinda modelin tahmin yapmasi zorlasir. 
# .. Bu yuzden X_train' i olabildigince fazla datayla egitmek onemlidir.

##### NLP Application with ML
# Classification of Tweets Data
# Bu calismada ML modellerinin NLP ile nasil kullanildigina dair ornekler yaptik. (NLP' de DL modelleri ML modellerine gore daha cok tercih edilir)
# The data Source: https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 50)

df = pd.read_csv("airline_tweets.csv")
df.head() 
# Havayolu sirketlerine ait atilan tweet' lerden olusan bir data. Datada yolcularin isimleri, yaptiklari yorumlar,
#  havayolu sirketlerinin adi, tarih gibi bilgiler mevcut. Target label' da neutral, positive, negative olmak uzere 
# .. 3 farkli class var. Butun NLP modellerinde sadece target ve text onemlidir, sadece bu ikisi ile islem yapacagiz. 
# .. Oncesinde ise datamizi biraz inceleyecegiz.

# ÖNEMLİ NOT: !!! NLP' de missing value' lar doldurulmaz, silinir. !!!

ax = sns.countplot(data = df, x ="airline", hue = "airline_sentiment")
for p in ax.containers:
    ax.bar_label(p)
# Hangi havayoluna kac sikayet gelmis grafigini cizdirdik. Cok yogun sikayet alan havayolu sirketleri var

ax = sns.countplot(data =df, x ="negativereason")
ax.bar_label(ax.containers[0])
plt.xticks(rotation =90);
# negativereason feature' ina gore bir grafik cizdirdik. Musterilerin cogu, musteri hizmetlerinden, gec ucuslardan vb. 
# .. sikayetci diyebiliriz :

ax = sns.countplot(data =df, x = "airline_sentiment")
ax.bar_label(ax.containers[0]);
# En fazla yorum negatif olarak yapilmis (Dengesiz bir dataseti)

df["airline_sentiment"].value_counts()

##### Cleaning Data
df2 = df.copy()
df2["text"].head()

##### Cleaning Data
# Cleaning islemi icin regex kutuphanesini kullanacagiz.
import re
s = "http\\:www.mynet.com #lateflight @airlines"
s = re.sub("http\S+", "", s).strip()
s
# re.sub --> Regexin bir fonksiyonudur. Asagida bu fonksiyon ile "http ile basla, bosluga kadar butun karakterleri 
# .. temizle." demis olduk. com' dan sonra gelen bosluga kadar her seyi temizledi, sonraki kisimlar kaldi. Cumlenin basinda 
# .. ve sonundaki bosluklari kaldirmak icin de strip() kullandik :
# "http\S+", "" : http ile baslayanlari al, yerine hicbir sey getirme.

s = re.sub("#\S+", "", s)
s
# @ ile baslayan tum ifadeleri bosluga kadar temizle :
# Yukarida temizledigimiz kelimeler her satirda oldugu icin modelde gürültüye sebep olur ve temizlenmeleri gerekir,
# .. egitime de bir katkilari yoktur.
# nltk.download('stopwords')
# nltk.download('stopwords')

# Sentimental analiz yapacagimiz icin not ve no kelimeleri analiz icin gerekli. for dongusu ile bu kelimeleri stopwords' ler 
# .. arasindan cikardik
stop_words = stopwords.words('english')
for i in ["not", "no"]:
        stop_words.remove(i)

# Tum cleaning islemlerini yapabilmek icin asagida cleaning isimli bir fonksiyon tanimladik.
# Yukarida tek tek yaptigimiz kaldirma islemlerini asagida ilk 3 satirda yaptik.
def cleaning(data):   
    import re  
    #1. Removing URLS
    data = re.sub('http\S+', '', data).strip()
    data = re.sub('www\S+', '', data).strip()
    #2. Removing Tags
    data = re.sub('#\S+', '', data).strip()
    #3. Removing Mentions
    data = re.sub('@\S+', '', data).strip()   
    #4. Removing upper brackets to keep negative auxiliary verbs in text
    data = data.replace("'", "")   
    #5. Tokenize
    text_tokens = word_tokenize(data.lower())   
    #6. Remove Puncs and number
    tokens_without_punc = [w for w in text_tokens if w.isalpha()]   
    #7. Removing Stopwords
    tokens_without_sw = [t for t in tokens_without_punc if t not in stop_words]  
    #8. lemma
    text_cleaned = [WordNetLemmatizer().lemmatize(t) for t in tokens_without_sw]  
    #joining
    return " ".join(text_cleaned)
# 4.islemde doesn't gibi kelimeleri ust ayractan kurtardik ki stopword uygulandiginda bu kelimeler corpus' ta bulunmaya devam etsin.
# 5.islemde tum kelimeleri kucuk harflere donusturerek tokenize ettik.
# 6.islemde noktalama isaretleri ve sayilari cikardik.
# 7.islemde corpusu stopword' lerden temizledik.
# 8.islemde tokenlerin köklerine indik.
# Son olarak join ile tüm tokenleri birlestirdik.

cleaning_text = df2["text"].apply(cleaning)
cleaning_text.head()
# Create ettigimiz fonksiyon icine df' teki 'text' sütununu apply fonksiyonu ile verdik ve cleaning islemlerini tamamladik

##### Features and Label
df2 = df2[["airline_sentiment", "text"]]
df2.head()
# Modelde kullanacagimiz iki sütunu aldik. text' in temizlenmemis halini aldik cunku asagida vectorizer fonksiyonu icinde bir
# .. parametre ile cleaning islemini yapacagiz :

##### Train Test Split
from sklearn.model_selection import train_test_split
X = df2["text"]
y= df2["airline_sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=101)
# X ve y' nin %20' sini test olarak ayirdik. Dengesiz bir dataseti oldugu icin stratify=y kullandik

##### Vectorization
# CountVectorizer islemine tabi tutulan corpus ile Logistic Regression, SVM, KNN, RF, AdaBoost modellerini kuracagiz, daha sonra 
# .. TF-IDF islemine tabi tutulan corpus ile yine tum modelleri kuracagiz ve en iyi skor aldigimiz modeli sececegiz.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(preprocessor=cleaning, min_df=3) # ngram_range=(1,2), max_features= 1500
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)
# Vectorizer Parametreleri :
# preprocessor --> Yukarida create ettigimiz fonksiyon olan cleaning' i bu parametreye tanimlarsak, cleaning islemini yapar.
# min_df=3 --> Corpus' ta 3 ve 3' ten az gecen tokenleri egitime dahil etme. (Egitime katkilari olmaz)
# max_features=150 --> Corpusta en fazla kullanilan ilk 1500 tokeni kullan. (Bunun yerine min_df kullanilmasi tavsiye edilir; 
# .. max_features kullanmak egitime engel olabilir, risklidir.)
# max_df --> En fazla kullanilan tokenlerin yuzde su kadarini al. (Kullanilmasi tavsiye edilmez. Farkinda olmadan egitime katkisi
# .. olacak tokenler cikarilabilir.)
# ngram_range(1,2) --> Cumledeki kelimeleri bir tek tek alir bir de ilk 2 kelimeyi alir, bir kaydirir, sonraki 2 kelimeyi alir 
# .. sona kadar bu islemi yaparak modeldeki kaliplari ogrenmeye calisir. Fakat bu sekilde de feature sayisi cok artacagi icin 
# .. egitim islemi cok uzar. Genel olarak (1,2) veya (1,3) olarak kullanilir, daha fazlasi tavsiye edilmez.
# .. (1,3) secildiginde (1,1), (1,2), (1,3)' u de yapar ve feature sayisi cok artar. 
# .. Guclu bir makine varsa bu parametre tercih edilebilir.

X_train_count.toarray()

pd.DataFrame(X_train_count.toarray(), columns = vectorizer.get_feature_names_out())
# CountVectorizer ve fit-transform islemlerinden sonra X_train' i array' e cevirdik ve token isimlerini alarak
# .. dataFrame' e donusturduk. 3126 tane feature' imiz var. Burada Feature Importance islemi yapamayiz fakat PCA yontemi
# .. kullanmak mantikli olur, o sekilde feature sayisi 100, 200 gibi bilesen sayisina dusurulebilir

##### Model Comparisons - Vectorization
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score
def eval(model, X_train, X_test):
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    print(confusion_matrix(y_test, y_pred))
    print("Test_Set")
    print(classification_report(y_test,y_pred))
    print("Train_Set")
    print(classification_report(y_train,y_pred_train))
    
##### Naive Bayes
# Naive Bayes teoremi daha cok DL modelleri ile kullanilir.
# P(A|B) --> A olayi gerceklestiginde B olayinin gerceklesme ihtimali nedir? Buradaki A olayi "don't like" olsun. don't like
# .. tokenleri birlikte kullanildiginda yorumun negatif olma olasiligi nedir durumuna Naive Bayes bakar. (P(don't like|negative)).
# .. Olasiliklar uzerinden calistigi icin NLP' de cok guzel sonuclar verir. Bu yuzden NLP ile kullanilmasi tavsiye edilir.
# Navie Bayes' te kullanilan 3 farkli algoritma var. Bunlardan en cok kullanilan iki tanesi : MultinomialNB, BernoulliNB
# MultinomialNB --> Daha cok multiclass datalarda tercih edilir.
# BernoulliNB --> Daha cok binary datalarda tercih edilir. (Dokumaninda her ikisinin de denenip hangisi iyi sonuc veriyorsa 
# .. onun secilmesi onerilir)    
from sklearn.naive_bayes import MultinomialNB, BernoulliNB # BernoulliNB for binary model
nb = MultinomialNB(alpha=3)
nb.fit(X_train_count,y_train)
# Bu datada multiclass ve MultinomialNB ile daha iyi sonuc verdigi icin bununla devam edecegiz.
# alpha --> Mesela ilk satirda 'able' tokeni hic gecmemis (0). O zaman olasilik hesabi yaparken P(0|positive)=0 cikacak.
# ..  alpha, bu tur tokenlere bir regularization islemi yapar. 'able' kelimesi gectigi zaman da corpus icindeki kullanim sıklıklarına
# .. gore dusuk de olsa mutlaka bir olasilik döndürür. Bu tokene dusuk bir agirlik verilmis olur fakat bu token cumlede geciyorsa
# ..  positive veya negative olma durumu ogrenilmis olur. Bu sayede Naive Bayes modellerin overfite gitmesi engellenir. 
# .. alpha degeri ne kadar buyurse o kadar yuksek derecede regularization islemi uygulanir (Ridge ve Lasso' daki gibi) (Default=1)
# Logistic Regression' da ci degeri, SVM' deki gama degeri kuculdukce uygulanan regularization islemi artiyordu, burda ise alpha degeri ne kadar buyurse uygulanan regularization islemi o kadar artar.
# !!! Eger bir overfit durumuyla karsilasilirsa yapilacak ilk islem, alpha degerini buyutmektir !!!

print("NB MODEL")
eval(nb, X_train_count, X_test_count)
# alpha=3 degeri ile train ve test datalarindaki negatif skorlarin birbirlerine yaklastigini gorduk, bu yuzden bu alpha degerini
# .. sectik (Negatif skorlar ile ilgileniyoruz). Train set ve Test set skorlari birbirine yakin, overfitting durumu yok

from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
scoring = {'accuracy': make_scorer(accuracy_score),
            'precision-neg': make_scorer(precision_score, average=None, labels=["negative"]),
            'recall-neg': make_scorer(recall_score, average=None, labels = ["negative"]),
            'f1-neg': make_scorer(f1_score, average=None, labels = ["negative"])}
model = MultinomialNB(alpha=3)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Negative skorlara yogunlasarak cross_validate islemi yaptik. Yukaridaki tek seferlik skorlar ile yakin skorlar elde ettik 

from yellowbrick.classifier import PrecisionRecallCurve
viz = PrecisionRecallCurve(
    MultinomialNB(alpha=3),   # model
    classes=nb.classes_,      # targettaki class' lar
    per_class=True,           # class isimlerini goster
    cmap="Set1"               # renklendirme
)
viz.fit(X_train_count,y_train)     # egitim yap
viz.score(X_test_count, y_test)    # skorlari al
viz.show();                        # gorsellestir
# Dengesiz bir datasetimiz oldugu icin precision_recall ile modelin skorlarina baktik. Negative class icin modelin
# .. genel performansi %93 

y_pred = nb.predict(X_test_count)
nb_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
nb_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
nb_AP_neg = viz.score_["negative"]
# Asagida modelleri bir tabloda karsilastiracagimiz icin; negative label icin recall_score, f1 score ve gorseldeki skoru(viz.score ile) alip 
# .. birer degiskene atadik

##### Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C = 0.02, max_iter=1000) #class_weight='balanced'
log.fit(X_train_count,y_train)
# C=1 default degeri ile model, overfite gittigi icin bu degeri kuculttuk. Degerleri manuel olarak denedik fakat GridSearch islemi de yapilabilirdik
# max_iter yetersiz kalirsa min noktaya ulasamadigi icin 'iterasyon sayisini' artir uyarisini verir, bu yuzden 1000 verdik.

print("LOG MODEL")
eval(log, X_train_count, X_test_count)
# Train ve test set skorlari birbirine yakin, overfit durumu yok

model = LogisticRegression(C = 0.02, max_iter=1000)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# CrossValidate skorlari tek seferlik skorlarla uyumlu 

viz = PrecisionRecallCurve(
    LogisticRegression(C = 0.02, max_iter=1000),
    classes=log.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test)
viz.show();
# PrecisionRecallCurve' de yukaridakine benzer bir skor elde ettik

y_pred = log.predict(X_test_count)
log_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
log_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
log_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

log = LogisticRegression(C = 0.02, max_iter=1000, class_weight='balanced')
log.fit(X_train_count,y_train)
# Modeldeki negative skorlarla ilgileniyoruz fakat yine de class_weight='balanced' dedik ve positive skorlar da yukseldi.
# .. positive yorumlar da onemli ise class_weight='balanced' mutlaka kullanilmali, cunku dengesiz bir datasetimiz var

print("LOG MODEL BALANCED")
eval(log, X_train_count, X_test_count)

##### SVM
from sklearn.svm import LinearSVC
svc = LinearSVC(C=0.01) # C=1 default degeri ile model overfite gittigi icin bu degeri asama asama kuculterek en iyi skoru aldigimiz 0.01' i sectik
svc.fit(X_train_count,y_train)

print("SVC MODEL")
eval(svc, X_train_count, X_test_count)

model = LinearSVC(C=0.01)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar dengeli

viz = PrecisionRecallCurve(
    LinearSVC(C=0.01),
    classes=svc.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test)
viz.show();
# Modelin genel performansi onceki modeller ile cok yakin

y_pred = svc.predict(X_test_count)
svc_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
svc_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
svc_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik 

##### KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_count,y_train)
# KNN icin Elbow metodu ile egitim cok uzun surecegi icin ve skorlar da cok kotu ciktigi icin birkac n_neighbors degeri deneyerek
# .. n_neighbors=7' de karar kildik ve skorlarimizi aldik 

print("KNN MODEL")
eval(knn, X_train_count, X_test_count)

model = KNeighborsClassifier(n_neighbors=7)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# CrossValidate sonucu alinan skorlar da oldukca kotu 

viz = PrecisionRecallCurve(
    KNeighborsClassifier(n_neighbors=7),
    classes=knn.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test)
viz.show();
# KNN modelin genel performansi onceki modellerden oldukca dusuk cikti

y_pred = knn.predict(X_test_count)
knn_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
knn_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
knn_AP_neg = viz.score_["negative"]
# Karsilastirma icin aldigimiz skorlari degiskenlere atadik

##### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(100, max_depth = 40, random_state = 42, n_jobs = -1) # class_weight="balanced"
rf.fit(X_train_count, y_train)
# GridSearch islemi yapmadik. RF modellerde ilk oynamamiz gereken parametre olan max_depth ile oynadik. 100, 200 gibi degerler verdigimizde
# .. modeli overfitten kurtaramadik; degerler kuculdukce negative skorlar icin overfitin engellendigini gorduk. En iyi skoru 40 degerinde aldik 

print("RF MODEL")
eval(rf, X_train_count, X_test_count)

model = RandomForestClassifier(100, max_depth = 40, random_state = 42, n_jobs = -1)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlarimiz ile CrossValidate sonucu aldigimiz skorlar tutarli

viz = PrecisionRecallCurve(
    RandomForestClassifier(100, max_depth = 40, random_state = 42, n_jobs = -1),
    classes=rf.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test)
viz.show();
# Negative class icin modelin genel performansi yuksek

y_pred = rf.predict(X_test_count)
rf_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
rf_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
rf_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

rf = RandomForestClassifier(100, max_depth = 40, random_state = 42, n_jobs = -1, class_weight="balanced")
rf.fit(X_train_count, y_train)
# RF modele bir de class_weight='balanced' ile baktik. positive' de train ve test setleri arasinda yuksek fark var. 
# .. Positive skorlara bakacak olsaydik, yukaridaki parametreler ile denemeler yapip modeli overfit durumundan kurtarmamiz gerekirdi

print("RF MODEL BALANCED")
eval(rf, X_train_count, X_test_count)

##### Ada Boost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators= 500, random_state = 42)
ada.fit(X_train_count, y_train)

print("Ada MODEL")
eval(ada, X_train_count, X_test_count)
# Ada Boost yerine XGBoost da tercih edilebilirdi. n_estimators= 500 ile train ve test skorlarinin birbirine yaklastigini gordugumuz icin
# .. bu degeri sectik.

model = AdaBoostClassifier(n_estimators= 500, random_state = 42)
scores = cross_validate(model, X_train_count, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirleriyle tutarli

viz = PrecisionRecallCurve(
    AdaBoostClassifier(n_estimators= 500, random_state = 42),
    classes=ada.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_count,y_train)
viz.score(X_test_count, y_test)
viz.show();
# Negative class icin modelin genel performansi yuksek 

y_pred = ada.predict(X_test_count)
ada_count_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
ada_count_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
ada_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

##### TF-IDF
# Yukarida CountVectorizer ile text' i sayisal verilere donusturup tum modeller icin skorlar aldik. Simdi ise sayisal verilere donusturme 
# .. islemini TF-IDF ile yapacagiz ve tum modeller icin yine skorlar alacagiz.
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vectorizer = TfidfVectorizer(preprocessor=cleaning, min_df=3)
X_train_tf_idf = tf_idf_vectorizer.fit_transform(X_train)
X_test_tf_idf = tf_idf_vectorizer.transform(X_test)
# preprocessor=cleaning diyerek create ettigimiz fonksiyon ile cleaning islemini yapmis olduk. (min_df=3 kullanmak idealdir.)

X_train_tf_idf.toarray()

pd.DataFrame(X_train_tf_idf.toarray(), columns = tf_idf_vectorizer.get_feature_names_out())

##### Model Comparisons TF-IDF
##### Naive Bayes
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
nb = MultinomialNB()
nb.fit(X_train_tf_idf,y_train)
# MultinomialNB() icindeki alpha parametresi ile oynamadik cunku default deger olan 1 ile negative class icin en iyi skoru verdi :

print("NB MODEL")
eval(nb, X_train_tf_idf, X_test_tf_idf)
# Modelde overfit durumu yok

model = MultinomialNB()
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# CrossValidate skorlari tek seferlik skorlar ile tutarli

viz = PrecisionRecallCurve(
    MultinomialNB(),
    classes=nb.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();
# Modelin genel performansı yüksek

y_pred = nb.predict(X_test_tf_idf)
nb_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
nb_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
nb_tfidf_AP_neg = viz.score_["negative"]

##### Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(C=0.4, max_iter=1000)
log.fit(X_train_tf_idf,y_train)
# C = 0.4 degeri ile model overfitten kurtuldugu icin bu degeri sectik.

print("LOG MODEL")
eval(log, X_train_tf_idf, X_test_tf_idf)

model = LogisticRegression(C=0.4, max_iter=1000)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar tutarli

viz = PrecisionRecallCurve(
    LogisticRegression(C=0.4, max_iter=1000),
    classes=log.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();
# Negative class icin modelin performansi yuksek

y_pred = log.predict(X_test_tf_idf)
log_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
log_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
log_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma icin skorlari degiskenlere atadik

log = LogisticRegression(C=0.4, max_iter=1000, class_weight="balanced")
log.fit(X_train_tf_idf,y_train)

print("LOG MODEL BALANCED")
eval(log, X_train_tf_idf, X_test_tf_idf)
# class_weight='balanced' ile skorlarimiza tekrar baktik. Positive class' in train ve test skorlari arasinda fark var, bu farki azaltmak 
# .. icin parametreler ile oynamak gerekir fakat bizim icin onemli class Negative class' i oldugu icin oynamayacagiz

##### SVM
from sklearn.svm import LinearSVC
svc = LinearSVC(C=0.1)
svc.fit(X_train_tf_idf,y_train)
# SVM modelde C=0.1 ile negative class' taki overfitting' i giderebildik.

print("SVC MODEL")
eval(svc, X_train_tf_idf, X_test_tf_idf)

model = LinearSVC(C=0.1)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakın

viz = PrecisionRecallCurve(
    LinearSVC(C=0.1),
    classes=svc.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();
# Negative class' a ait genel performans yuksek

y_pred = svc.predict(X_test_tf_idf)
svc_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
svc_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
svc_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma yapmak uzere skorlari degiskenlere atadik

##### KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_tf_idf,y_train)

print("KNN MODEL")
eval(knn, X_train_tf_idf, X_test_tf_idf)
# KNN modeli skorlari onceki gibi cok kotu oldugu icin parametreler ile oynamanin da bir anlami yok

model = KNeighborsClassifier(n_neighbors=7)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Modelin genel performansi cok dusuk

viz = PrecisionRecallCurve(
    KNeighborsClassifier(n_neighbors=7),
    classes=knn.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();

y_pred = knn.predict(X_test_tf_idf)
knn_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
knn_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
knn_tfidf_AP_neg = viz.score_["negative"]
# Karsilastirma yapmak uzere skorlari degiskenlere atadik

##### RandomForest
rf = RandomForestClassifier(100, max_depth=40, random_state = 42, n_jobs = -1)
rf.fit(X_train_tf_idf, y_train)
# Parametreler ile oynayarak en iyi skoru aldigimiz parametreleri sectik

print("RF MODEL")
eval(rf, X_train_tf_idf, X_test_tf_idf)

model = RandomForestClassifier(100, max_depth=40, random_state = 42, n_jobs = -1)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakin :

viz = PrecisionRecallCurve(
    RandomForestClassifier(100, max_depth=40, random_state = 42, n_jobs = -1),
    classes=rf.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();
# Negative class' a ait genel performans yuksek

y_pred = rf.predict(X_test_tf_idf)
rf_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
rf_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
rf_tfidf_AP_neg = viz.score_["negative"]

rf = RandomForestClassifier(100, max_depth=15, random_state = 42, n_jobs = -1, class_weight="balanced")
rf.fit(X_train_tf_idf, y_train)

print("RF MODEL BALANCED")
eval(rf, X_train_tf_idf, X_test_tf_idf)
# class_weight="balanced" secildiginde positive skorlar birbirine biraz daha yaklasti 

##### Ada Boost
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators= 500, random_state = 42)
ada.fit(X_train_tf_idf, y_train)

print("Ada MODEL")
eval(ada, X_train_tf_idf, X_test_tf_idf)
# Negative class' lar icin en dengeli skoru AdaBoost modelde aldik, precision ve recall skorlari birbirlerine cok yakin. Musteri dengeli bir 
# .. skor istediginde bu model sunulabilir :

model = AdaBoostClassifier(n_estimators= 500, random_state = 42)
scores = cross_validate(model, X_train_tf_idf, y_train, scoring = scoring, cv = 10)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]
# Tek seferlik skorlar ile CrossValidate sonucu alinan skorlar birbirine yakin

viz = PrecisionRecallCurve(
    AdaBoostClassifier(n_estimators= 500, random_state = 42),
    classes=ada.classes_,
    per_class=True,
    cmap="Set1"
)
viz.fit(X_train_tf_idf,y_train)
viz.score(X_test_tf_idf, y_test)
viz.show();
# precision ve recall degerleri diger modellere gore daha dusuk ciktigi icin genel performans da %90 cikmis

y_pred = ada.predict(X_test_tf_idf)
ada_tfidf_rec_neg = recall_score(y_test, y_pred, labels = ["negative"], average = None)
ada_tfidf_f1_neg = f1_score(y_test, y_pred, labels = ["negative"], average = None)
ada_tfidf_AP_neg = viz.score_["negative"]

##### Compare Scoring
compare = pd.DataFrame({"Model": ["NaiveBayes_count", "LogReg_count", "SVM_count", "KNN_count", "Random Forest_count", 
                                  "AdaBoost_count", "NaiveBayes_tfidf", "LogReg_tfidf", "SVM_tfidf", "KNN_tfidf", 
                                  "Random Forest_tfidf", "AdaBoost_tfidf"],
                        
                        "F1_Score_Negative": [nb_count_f1_neg[0], log_count_f1_neg[0], svc_count_f1_neg[0], knn_count_f1_neg[0],
                                             rf_count_f1_neg[0], ada_count_f1_neg[0], nb_tfidf_f1_neg[0], log_tfidf_f1_neg[0],
                                             svc_tfidf_f1_neg[0], knn_tfidf_f1_neg[0], rf_tfidf_f1_neg[0], ada_tfidf_f1_neg[0]],
                        
                        "Recall_Score_Negative": [nb_count_rec_neg[0], log_count_rec_neg[0], svc_count_rec_neg[0], 
                                                  knn_count_rec_neg[0], rf_count_rec_neg[0], ada_count_rec_neg[0], 
                                                  nb_tfidf_rec_neg[0], log_tfidf_rec_neg[0], svc_tfidf_rec_neg[0], 
                                                  knn_tfidf_rec_neg[0], rf_tfidf_rec_neg[0], ada_tfidf_rec_neg[0]],
                        
                        "Precision_Recall_Score_Negative": [nb_AP_neg, log_AP_neg, svc_AP_neg, knn_AP_neg, rf_AP_neg,
                                                          ada_AP_neg, nb_tfidf_AP_neg, log_tfidf_AP_neg, svc_tfidf_AP_neg,
                                                           knn_tfidf_AP_neg, rf_tfidf_AP_neg, ada_tfidf_AP_neg]})

def labels(ax):
                        
    for p in ax.patches:
        width = p.get_width()                        # get bar length
        ax.text(width,                               # set the text at 1 unit right of the bar
                p.get_y() + p.get_height() / 2,      # get Y coordinate + X coordinate / 2
                '{:1.3f}'.format(width),             # set variable to display, 2 decimals
                ha = 'left',                         # horizontal alignment
                va = 'center')                       # vertical alignment
    
plt.figure(figsize=(15,30))
plt.subplot(311)
compare = compare.sort_values(by="Recall_Score_Negative", ascending=False)
ax=sns.barplot(x="Recall_Score_Negative", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(312)
compare = compare.sort_values(by="F1_Score_Negative", ascending=False)
ax=sns.barplot(x="F1_Score_Negative", y="Model", data=compare, palette="Blues_d")
labels(ax)

plt.subplot(313)
compare = compare.sort_values(by="Precision_Recall_Score_Negative", ascending=False)
ax=sns.barplot(x="Precision_Recall_Score_Negative", y="Model", data=compare, palette="Blues_d")
labels(ax)
plt.show();

# Tum modellerden elde ettigimiz skorlari kiyaslamak amaciyla bir fonksiyon create ettik.
# compare degiskeni icine olusturdugumuz tum model isimlerini tanimladik. Bununla birlikte yukarida negative class' i icin aldigimiz tum f1 score, 
# .. recall score ve precision recal score' lari da tanimladik. (recall ve f1 score, precision hakkinda da inside sagladigi icin onu yazmadik)
# nb_count_f1_neg[0] : Skorlari array' den kurtarmak icin [0] indexlemesini yapiyoruz.
# Tanimladigimiz fonksiyon ile 3 ayri grafik elde ettik; tum modeller icin ilk grafik recall, ikinci grafik f1 score, ucuncu grafik ise precision 
# .. recall skorlarini temsil ediyor.
# recall grafiginde en yuksek skoru RF modeli verdi.
# f1 score grafiginde yuksek skorlari SVM modelleri verdi fakat LogReg_tfidf ile de arasinda cok fazla bir fark yok. LogReg_tfidf' in recall skoru 
# .. da %95. RF model recall grafiginde en yuksek skoru vermisti fakat f1 grafiginde skorlari dusuk. Hem recall hem f1 skorunun yuksek olmasini, 
# .. bununla birlikte modelin genel performansinin da yuksek olmasini istiyoruz.
# Hem hizli calistigi icin hem de f1 skoru yuksek oldugu icin LogReg_tfidf secmek daha mantikli.
# LogReg_tfidf' in genel performansina baktigimizda %92, en yuksek skor ile arasinda cok fazla bir fark yok. Model olarak LogReg_tfidf' i secmeye
# .. karar verdik.

##### For New Tweets, prediction with pipeline
# Kullanmaya karar verdigimiz modelimiz ile prediction yapacagiz.
from sklearn.pipeline import Pipeline
pipe = Pipeline([('tfidf',TfidfVectorizer(preprocessor=cleaning, min_df=3)),('log',LogisticRegression(C=0.4, max_iter=1000))])
# Pipeline ile fit_transform ve fit_prediction islemlerini yapabiliyorduk. Ilk yazdigimiz fonksiyon fit_transform,
# ..  ikinci yazdigimiz ise fit_prediction islemlerini yapar.
# En iyi islemin TF-IDF olduguna karar vermistik, ilk olarak bunu tanimladik, ikinci kisma Logistic Regression' da en iyi sonucu 
# .. aldigimiz parametreleri tanimladik

pipe.fit(X, y)
# Pipe icine X ve y' nin temizlenmemis halini veriyoruz, cunku TfidfVectorizer icine tanimladigimiz cleaning fonksiyonu bu islemi yapacak.
# .. Train ve test olarak ayirmadan tum corpus' u modele verdik

# Yukaridaki kodlar ile modelin egitim islemleri tamamlandi. Olusturdugumuz bu model ile prediction' lar alacagiz.
# NOTE: !!! Modelden bir prediction alabilmek icin sample' i mutlaka series' e donusturmek gerekir. !!!
tweet = "it was not the worst flight i have ever been"
tweet = pd.Series(tweet)
pipe.predict(tweet)  # Output : array(['negative'], dtype=object)
# Prediction icin verilen sample' da noktalama isareti veya ozel karakterler olsa bile pipe bu karakterleri temizleyecektir, ayrica 
# .. temizlememize gerek yok.
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "didn't enjoy flight"
tweet = pd.Series(tweet)
pipe.predict(tweet) # array(['negative'], dtype=object)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "it is amazing"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in positive bir yorum oldugunu bildi

tweet = "don't enjoy flight  at all"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "I don't think I'll ever use American Airlines any more"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz tweet' in negative bir yorum oldugunu bildi

tweet = "it isn't amazing"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz  tweet' in negative yorum oldugunu bilemedi. Bir sonraki adimda bunun nedenini arastiracagiz

tweet = "I don't love the flight"
tweet = pd.Series(tweet)
pipe.predict(tweet)
# Modelimiz  tweet' in negative yorum oldugunu bilemedi. Bir sonraki adimda bunun nedenini arastiracagiz

##### Collect Words and Counting words
tweets = cleaning_text
tweets
# cleaning_text, yukarida CountVectorizer ile degil manuel olarak temizledigimiz text idi. Bir sonraki kodda bu yorumlarin hepsini 
# .. join ile birlestirdik ve all_words degiskenine atadik

all_words = " ".join(tweets)
all_words[:100]    # Ilk 100 kelime, join ile birlesmis halde.

counter = Counter(word_tokenize(all_words))
# Counter() : Corpus' ta gecen tokenleri teker teker sayar. counter.most_common() ile de bunlarin kacar tane oldugunu en fazladan en aza 
# .. dogru siralar.
# word_tokenize : Corpus icindeki tum kelimeleri token haline getirir.

counter.most_common()

# 'enjoy', 'love', 'like' tokenlerinin corpus' ta kacar kere kullanildigini for dongusu ile bulalım
for i in counter.most_common():
    if "enjoy" == i[0]:
        print(i)

for i in counter.most_common():
    if "love" == i[0]:
        print(i)

for i in counter.most_common():
    if "like" == i[0]:
        print(i)

# Modelimiz yukarida 'dont love' in negatif bir yorum oldugunu bilememisti. Asagidaki for dongusunde; tweets icinde 'love' ve 'dont'
# .. kelimelerinin birlikte gectigi ve label' i ' negative olan olan kac cumle varsa dondur dedik. Bu sekilde 5 cumle varmis ve sayisi az 
# .. oldugu icin egitim icin yetersiz kalmis diyebiliriz. Asagida modelin tahmin yapmasi icin yetersiz kalan diger ornekler var
counter = 0
for i,j in enumerate(tweets):
    if "love" in j and "dont" in j and y[i]=="negative":
        counter += 1
print(counter)      

counter = 0
for i,j in enumerate(tweets):
    if "like" in j and "dont" in j and y[i]=="negative":
        counter += 1
print(counter)

counter = 0
for i,j in enumerate(tweets):
    if "like" in j and "didnt" in j and y[i]=="negative":
        counter += 1
print(counter)

counter = 0
for i,j in enumerate(tweets):
    if "amazing" in j and "wasnt" in j and y[i]=="negative":
        counter += 1
print(counter)

counter = 0
for i,j in enumerate(tweets):
    if "love" in j and y[i]=="neutral":
        counter += 1
print(counter)

##### WordCloud - Repetition of Words
# Bu asamada corpus' ta siklikla kullanilmis olan tokenleri gorsellestirecegiz. Bunun icin wordcloud kutuphanesini import edecegiz.
##### Collect Words
all_words = " ".join(tweets) # Bir onceki adimda yaptigimiz gibi corpus' tan elde ettigimiz tum kelimeleri birlestirdik

all_words[:100] # Ilk 100 karakterine bakarak yorumlarin birlestigini teyit ettik

##### Create Word Cloud
from wordcloud import WordCloud
worldcloud = WordCloud(background_color="white", max_words =250)
# background_color : Arka plan rengini ayarlar.
# max_words = 250 : En fazla kullanilan 250 kelimeyi goster. (Baska sayilar verilebilir)

worldcloud.generate(all_words) #  generate ile en fazla kullanilan 250 kelimeye ait gorselin yapisi arka planda olusturulur.

import matplotlib.pyplot as plt
plt.figure(figsize = (13,13))
plt.imshow(worldcloud, interpolation="bilinear",)                 # interpolation : renklendirme
plt.axis("off")                                                   # Cerceve olsun mu?
plt.show()
# Gorsellestirme icin matplotlib altyapisini kullandik. imshow icine olusturdugumuz wordcloud' i verince corpus icinde daha cok gecen 
# .. ifadeler daha buyuk harfli olarak gorselestirilmis oldu

#%% NLP-3
###### Word Embedding Yöntemi
# Kelimeler arası anlamsal ilişkileri gösteren sayısal vektörlerdir
# CountVektorizer ve TF-IDF tokenlerin kendi aralarındaki anlamsal ilişkiyi yakalayamaz. Mesela
# .. güzel-çirkin arasındaki anlamsal ilişkiyi yakalayamaz
# Word Embedding tokenlerin kendi aralarındaki anlamsal ilişkiyi yakalar. Bu yüzden tercih edilir
# İki kelimenin veya cmlenin birbirine ne kadar yakın anlamda olduğun Cosinus similarity fonksiyonu ile bakılabilir
# Cosinus similarity değeri 1'e ne kadar yakınsa bu kelimeler anlamsal olarak birbirlerine o kadar yakın demektir

# Vektörün boyutunun ne olacağına biz karar veriyoruz. Piyasada genelde 50,100,300 boyutlu vektörler tercih edilir
"""
Gender: Man   = -1   Demekki gender ile %100 ilişkili bu iki kelime ama birbirlerinin tam zıttı
         Woman =  1

        King   = -0.95  Model, King-Queen'in anlamsal olarak birbirine yakın olduğunu fakat gender özelliği üzerinden de birbirlerinin zıttı
        Queen  = 0.07   .. olduğunu anlamış

        Apple  = 0.00  Model, bunların gender ile bir ilgisi olmadığını anlamış
        Orange = 0.01

Royal: Man   = 0.01    Model, bunların royal ile bir ilgisi olmadığını anlamış
       Woman = 0.02

       King   = -0.95  Model, King-Queen'in kraliyetle direk ikişili olduğunu anlamış bu yüzden
        Queen  = 0.07  .. zıt işaretler yok

.......

Age:   King = 0.7     King-Queen olabilmek için belli bir yaşa gelmek gerekir. Bu yüzden age ile
       Queen = 0.69   .. güçlü bir ilişki var

.....

Food: ......
"""

# Tokenler arasında anlamsal bir ilişki olup olmadığı şu şekilde hesaplanır
# Örneğin Man-Woman arasında anlamsal bir ilişki var mı diye bakacağız. Man vektöründen, woman vektörü çıkartılır
"""
Man    Woman
-1   -    1      = -2
0.01 -  0.02     = -0.01
0.03 -  0.02     =  0.01
0.09 -  0.01     =  0.08
# Farkları 0'a yakın olan feature ların birbirleriyle çok benzer olduğunu model anlıyor
"""

# Matematiksel olarak da bu işlemi şu şekilde yapar. ;
"""
Mesela bütün feature'lar 100 boyutlu olsun.  Man ve Woman 100 boyutlu vektörleri şu vektörlerle temsil edilsin.
.. (PCA'de birçok feature'ı .. temsil eden bir bileşen gibi)
.. Alttaki 2 vektör arasındaki açı 10 derece olsun

  Man     Woman
   |     /
   |    /
   |   /
   |  /
   | /
   |/

Bu açının cosinüs ü alınır. Cos10.
Benzerlik belirlenirken arka planda cosinus similarity işlemi yapılır. aradaki açı ne kadar dar
.. ise anlamsal olarak birbirlerine o kadar yakınlar demektir. Aradaki açı 0 ise bu iki kelime
.. AYNI kelime demektir
"""
# Anlamsal ilişkili olan kelimelerin vektörel uzaklıklarının yanında öklid uzaklıkları da yakındır

#### Cosine Similarity
# İki veya daha fazla vektör arasındaki benzerliği ölçer

"""
   y      A      B
   |     |     /
   |    |     /
   |   |(10)/               --> Cos(10) = 0.9848 --> %98 similar
   |  |   /
   | |  /
   ||/____________ x

Benzerliği sözlükteki anlama göre değil, birlikte kullanılma sıklığına göre veya aynı eylemi yapma
.. durumlarına göre kurar
"""

##### Word Embedding Algorithms
# 1. Embedding Layer
# 2. Word2 Vec
# 3. Global Vectors(Glove)
# 4. Embedding from Language Models(ELMO)
# 5. Bidirectional Encoder Representations from Transformers(BERT)

##### 1.Embedding Layer:
# ....
# ...
# To be continued...





















