# Spectrum Analysis EP
Autorzy
Wojciech Żurawski – https://github.com/wzurawski015 watwzwp@gmail.com
Eugeniusz Pazderski

**Spectrum Analysis EP** to program napisany w Pythonie, który analizuje funkcje autokorelacyjne i oblicza widma mocy za pomocą szybkiej transformaty Fouriera (FFT). Program jest w stanie przetwarzać wiele plików danych jednocześnie, wykluczając określone pliki na podstawie listy wykluczeń. Wyniki analizy są przedstawiane w formie wykresów statycznych i interaktywnych oraz zawarte w szczegółowym raporcie HTML.

## Spis treści

- [Opis](#opis)
- [Wymagania](#wymagania)
- [Instalacja](#instalacja)
- [Struktura projektu](#struktura-projektu)
- [Przygotowanie danych](#przygotowanie-danych)
  - [Format danych wejściowych](#format-danych-wejściowych)
  - [Plik wykluczeń](#plik-wykluczeń)
- [Użycie](#użycie)
- [Przykładowe wyniki](#przykładowe-wyniki)
- [Raport](#raport)
- [Licencja](#licencja)
- [Autorzy](#autorzy)

## Opis

**Spectrum Analysis EP** to program, który:

- Wczytuje i analizuje funkcje autokorelacyjne z wielu plików danych.
- Wyklucza określone pliki z analizy na podstawie listy wykluczeń.
- Symetryzuje funkcje autokorelacyjne w celu uzyskania rzeczywistego widma mocy.
- Usuwa składową stałą (DC offset) z funkcji autokorelacyjnych.
- Wykonuje szybką transformatę Fouriera (FFT) na funkcjach autokorelacyjnych.
- Oblicza widmo mocy i przekształca je na skalę decybelową (dB).
- Generuje wykresy funkcji autokorelacyjnych i widm mocy w formie statycznej (`.png`) oraz interaktywnej (`.html`).
- Tworzy szczegółowy raport w formacie HTML zawierający wszystkie wyniki analizy.

## Wymagania

- Python 3.6 lub nowszy
- Pakiety wymienione w `requirements.txt`:
  - `numpy`
  - `scipy`
  - `matplotlib`
  - `plotly`

## Instalacja

1. **Sklonuj repozytorium:**

   ```bash
   git clone https://github.com/wzurawski015/spectrum_analysis_ep.git

Struktura projektu
spectrum_analysis_ep/
├── autofft.py               # Zaktualizowany skrypt programu
├── requirements.txt         # Plik z zależnościami
├── README.md                # Dokumentacja projektu
├── LICENSE                  # Informacje o licencji
├── .gitignore               # Plik konfiguracji Git
├── data/                    # Folder z danymi wejściowymi
│   ├── input1.dat            # Przykładowy plik danych
│   ├── input2.dat            # Kolejny plik danych
│   ├── sample3.dat           # Inny plik danych
│   └── exclude               # Plik z nazwami plików do wykluczenia (opcjonalny)
└── output/                  # Folder z wynikami (generowany automatycznie)

Przygotowanie danych
Format danych wejściowych
Pliki z danymi wejściowymi powinny zawierać cztery funkcje autokorelacyjne, każda reprezentowana przez pary:
Nr_kanalu  wartosc_funkcji

Nr_kanalu: Numer kanału (lag/opóźnienie), liczba całkowita.
wartosc_funkcji: Wartość funkcji autokorelacyjnej dla danego numeru kanału, liczba zmiennoprzecinkowa.
Wymagania dotyczące danych:

Numery kanałów powinny być od 0 do 16387.
Dane dla każdej funkcji powinny być umieszczone kolejno, bez przerw między nimi.
Każda funkcja autokorelacyjna powinna zawierać dokładnie 4097 próbek dla funkcji 1-3 i 4096 próbek dla funkcji 4.

Przykład fragmentu pliku input1.dat:
0       124413044
1       95279359
2       64397895
...
16387   62230859

Plik wykluczeń
Aby wykluczyć określone pliki z analizy, utwórz plik exclude w katalogu data i dodaj do niego nazwy plików, które mają być pominięte. Każda nazwa pliku powinna być na osobnej linii.

Przykład zawartości pliku data/exclude:
input2.dat
sample3.dat
old_data.dat

Uwagi:
Nazwy plików muszą dokładnie odpowiadać nazwom plików w katalogu data.
Nie uwzględniaj ścieżek, tylko same nazwy plików.
Użycie
Przygotuj dane wejściowe:

Umieść wszystkie pliki z danymi wejściowymi w folderze data.
Jeśli chcesz wykluczyć niektóre pliki, utwórz plik data/exclude z listą nazw plików do pominięcia.
Uruchom program:

python autofft.py

Sprawdź wyniki:

Wyniki zostaną zapisane w folderze output:
Pliki z funkcjami autokorelacyjnymi (*.src).
Pliki z widmami mocy (*.fft).
Wykresy funkcji autokorelacyjnych (*_autocorr.png).
Wykresy widma mocy (*.png).
Interaktywne wykresy widma mocy (*_interactive.html).
Raport (raport.html).
Przykładowe wyniki
Po uruchomieniu programu w folderze output znajdziesz:

Pliki z funkcjami autokorelacyjnymi (input1_pcal1.src, input1_pcal2.src, ...):
Zawierają przetworzone i symetryzowane funkcje autokorelacyjne.
Pliki z widmami mocy (input1_pcal1.fft, input1_pcal2.fft, ...):
Zawierają widma mocy w skali logarytmicznej (dB).
Wykresy funkcji autokorelacyjnych (input1_pcal1_autocorr.png, ...):
Przedstawiają graficzną interpretację funkcji autokorelacyjnych.
Wykresy widma mocy (input1_pcal1.png, ...):
Przedstawiają graficzną interpretację widm mocy.
Interaktywne wykresy widma mocy (input1_pcal1_interactive.html, ...):
Umożliwiają interaktywną eksplorację widma mocy w przeglądarce internetowej.
Raport HTML (raport.html):
Zawiera wszystkie powyższe wyniki w jednym, przejrzystym pliku HTML.
Raport
Raport HTML (raport.html) zawiera:

Nagłówek z tytułem i datą generacji raportu.
Sekcje dla każdego przetworzonego pliku danych:
Nazwa pliku danych.
Wykres funkcji autokorelacyjnej.
Wykres widma mocy.
Link do interaktywnego wykresu widma mocy.
Przykład sekcji dla jednego pliku danych:

<div class="file-section">
    <h2>Plik danych: input1.dat</h2>
    <h3>Funkcja autokorelacyjna 1 z pliku input1.dat</h3>
    <p><strong>Funkcja autokorelacyjna 1:</strong></p>
    <img src="input1_pcal1_autocorr.png" alt="Autokorelacja 1" width="800">
    <p><strong>Widmo mocy 1:</strong></p>
    <img src="input1_pcal1.png" alt="Widmo mocy 1" width="800">
    <p><strong>Interaktywny wykres widma mocy 1:</strong> <a href="input1_pcal1_interactive.html" target="_blank">Otwórz</a></p>
    <hr>
</div>

Licencja
Ten projekt jest objęty licencją MIT License – więcej informacji w pliku LICENSE.

Autorzy
Wojciech Żurawski – GitHub
Eugeniusz Pazderski

Dodatkowe Uwagi
Częstotliwość próbkowania FS:

Upewnij się, że wartość FS w skrypcie autofft.py odpowiada częstotliwości próbkowania Twoich danych. Domyślnie jest ustawiona na 1000 Hz. Jeśli Twoje dane mają inną częstotliwość próbkowania, zmień tę wartość na odpowiednią.
FS = 1000  # Częstotliwość próbkowania w Hz

Rozszerzenia plików danych:

Skrypt jest skonfigurowany do analizowania wszystkich plików w katalogu data niezależnie od ich rozszerzenia. Jeśli chcesz ograniczyć analizę do określonych rozszerzeń (np. .dat, .txt), zmodyfikuj linię w funkcji main():
data_files = glob.glob(os.path.join(DATA_DIR, '*'))

Na przykład, aby analizować tylko pliki .dat:

data_files = glob.glob(os.path.join(DATA_DIR, '*.dat'))

Obsługa błędów:

Skrypt informuje użytkownika o wszelkich problemach z danymi, takich jak nieprawidłowa liczba wierszy czy błędy podczas zapisywania plików. Ułatwia to diagnozowanie i rozwiązywanie problemów.

Dodawanie dodatkowych funkcjonalności:

Jeśli chcesz dodać więcej szczegółów do raportu, takich jak statystyki opisowe, histogramy czy inne rodzaje wykresów, możesz rozszerzyć funkcję generate_report() oraz dodać odpowiednie funkcje do przetwarzania i wizualizacji danych.

**Dziękujemy za skorzystanie z Spectrum Analysis EP! Jeśli masz pytania lub sugestie dotyczące projektu, zapraszamy do kontaktu poprzez zgłoszenie problemu (issue) na GitHubie lub bezpośredni kontakt z autorami.


