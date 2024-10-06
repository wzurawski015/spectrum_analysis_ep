import os
import glob
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.fft import fft
from datetime import datetime

# Stała: częstotliwość próbkowania w Hz
FS = 1000

# Domyślne ścieżki do katalogów i plików
DATA_DIR_DEFAULT = 'data'  # Katalog, gdzie znajdują się dane wejściowe
EXCLUDE_FILE_DEFAULT = os.path.join(DATA_DIR_DEFAULT, 'exclude')  # Plik zawierający listę plików do wykluczenia z analizy
OUTPUT_DIR_DEFAULT = 'output'  # Katalog, gdzie zostaną zapisane wyniki

def setup_logging():
    """
    Konfiguruje system logowania, który zapisuje komunikaty zarówno do pliku jak i wyświetla je w konsoli.
    Logowanie ułatwia śledzenie przebiegu analizy oraz diagnozowanie problemów.
    """
    logging.basicConfig(
        level=logging.INFO,  # Ustawienie poziomu logowania na "INFO" oznacza, że będą logowane istotne komunikaty
        format='%(asctime)s - %(levelname)s - %(message)s',  # Format logów: czas - poziom logowania - komunikat
        handlers=[
            logging.FileHandler("spectrum_analysis.log"),  # Zapis logów do pliku
            logging.StreamHandler()  # Wyświetlanie logów na ekranie (konsoli)
        ]
    )

def parse_arguments():
    """
    Parsuje argumenty linii poleceń, co pozwala na dostosowanie parametrów analizy przy jej uruchomieniu.
    Przykładowo, można podać inny katalog z danymi lub zmienić częstotliwość próbkowania.
    """
    parser = argparse.ArgumentParser(description='Spectrum Analysis EP')  # Opis programu dla użytkownika
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT, help='Katalog z danymi wejściowymi')
    parser.add_argument('--exclude_file', type=str, default=EXCLUDE_FILE_DEFAULT, help='Plik z listą wykluczeń')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR_DEFAULT, help='Katalog z wynikami')
    parser.add_argument('--fs', type=int, default=FS, help='Częstotliwość próbkowania w Hz')
    return parser.parse_args()  # Zwraca obiekt z argumentami przekazanymi podczas uruchamiania

def load_exclude_list(exclude_path):
    """
    Wczytuje listę plików do wykluczenia z analizy na podstawie pliku tekstowego.
    Jeśli plik z listą wykluczeń nie istnieje, zwraca pustą listę.
    """
    exclude_files = []  # Pusta lista na wykluczenia
    if os.path.exists(exclude_path):  # Sprawdzenie, czy plik z wykluczeniami istnieje
        with open(exclude_path, 'r') as f:  # Otwieramy plik z wykluczeniami
            exclude_files = [line.strip() for line in f if line.strip()]  # Wczytujemy każdą linię pliku
        logging.info(f"Załadowano {len(exclude_files)} plików do wykluczenia.")  # Informacja o liczbie wykluczeń
    else:
        logging.info("Plik 'exclude' nie istnieje. Analizowane będą wszystkie pliki w katalogu 'data'.")
    return exclude_files

def get_data_files(data_dir, exclude_files):
    """
    Pobiera listę plików do analizy, wykluczając pliki znajdujące się na liście wykluczeń.
    Zwraca tylko te pliki, które nie znajdują się na liście oraz nie są plikiem 'exclude'.
    """
    all_files = glob.glob(os.path.join(data_dir, '*'))  # Pobranie wszystkich plików z katalogu 'data'
    data_files = [
        f for f in all_files
        if os.path.isfile(f)  # Sprawdzamy, czy to plik (a nie np. folder)
        and os.path.basename(f) not in exclude_files  # Sprawdzamy, czy plik nie jest na liście wykluczeń
        and os.path.basename(f) != 'exclude'  # Pomijamy sam plik 'exclude'
    ]
    logging.info(f"Znaleziono {len(data_files)} plików do analizy.")  # Informacja o liczbie plików do analizy
    return data_files

def read_autocorrelation(file_path):
    """
    Wczytuje funkcje autokorelacyjne z pliku danych.
    Każda funkcja autokorelacyjna jest podzielona na 4 części (pierwsze trzy mają 4097 próbek, a czwarta 4096).
    """
    try:
        # Odczytujemy wszystkie dane z pliku jako macierz (każdy wiersz to kanał i wartość)
        data = np.loadtxt(file_path)
        channels = data[:, 0].astype(int)  # Pierwsza kolumna zawiera numery kanałów
        values = data[:, 1].astype(float)  # Druga kolumna zawiera wartości autokorelacji

        # Podział wartości na cztery funkcje autokorelacyjne
        autocorr1 = values[0:4097]
        autocorr2 = values[4097:8194]
        autocorr3 = values[8194:12291]
        autocorr4 = values[12291:16387]
        return [autocorr1, autocorr2, autocorr3, autocorr4]
    except Exception as e:
        logging.error(f"Błąd podczas wczytywania pliku {file_path}: {e}")  # Zapis błędu w logu
        return []  # Zwraca pustą listę w przypadku błędu

def symmetrize(autocorr):
    """
    Symetryzuje funkcję autokorelacyjną. Symetryzacja jest potrzebna, by uzyskać poprawne widmo mocy.
    Funkcja autokorelacyjna powinna być symetryczna względem zera.
    """
    return np.concatenate((autocorr, autocorr[::-1]))  # Łączymy funkcję z jej odbiciem lustrzanym

def remove_dc_offset(autocorr):
    """
    Usuwa składową stałą (DC offset) z funkcji autokorelacyjnej, co pozwala uniknąć zniekształceń w analizie widma.
    Składowa stała to wartość średnia funkcji.
    """
    return autocorr - np.mean(autocorr)  # Odejmujemy średnią wartość z funkcji autokorelacyjnej

def compute_fft(autocorr):
    """
    Wykonuje szybką transformatę Fouriera (FFT) na funkcji autokorelacyjnej, aby przejść z domeny czasowej do częstotliwościowej.

    FFT (Fast Fourier Transform) to algorytm służący do szybkiego przekształcenia sygnału z domeny czasu do domeny częstotliwości.
    Wynik FFT zawiera częstotliwości składowe sygnału oraz ich amplitudy.
    """
    fft_result = fft(autocorr)  # Obliczamy FFT
    return np.abs(fft_result)  # Zwracamy moduł (wartości bezwzględne) wyników FFT

def compute_power_spectrum(fft_result):
    """
    Oblicza widmo mocy na podstawie wyników FFT i przekształca wartości na skalę decybelową (dB).
    Skalowanie logarytmiczne (dB) jest używane, ponieważ wartości mocy mogą mieć bardzo szeroki zakres.

    Widmo mocy opisuje, jak energia sygnału jest rozłożona na różne częstotliwości.
    """
    power_spectrum = 10 * np.log10(fft_result + 1e-12)  # Mała wartość 1e-12 dodawana jest dla stabilności logarytmu
    return power_spectrum

def save_intermediate_results(output_prefix, autocorr, sym_autocorr, cleaned_autocorr, fft_result, power_spectrum):
    """
    Zapisuje pośrednie wyniki obliczeń (np. funkcje autokorelacyjne, FFT) do plików NumPy (.npy).
    Pliki te mogą być później wykorzystane do bardziej szczegółowej analizy.
    """
    np.save(f"{output_prefix}_autocorr.npy", autocorr)  # Zapisujemy oryginalną funkcję autokorelacyjną
    np.save(f"{output_prefix}_sym_autocorr.npy", sym_autocorr)  # Zapisujemy symetryzowaną funkcję
    np.save(f"{output_prefix}_cleaned_autocorr.npy", cleaned_autocorr)  # Zapisujemy funkcję po usunięciu DC offset
    np.save(f"{output_prefix}_fft_result.npy", fft_result)  # Zapisujemy wyniki FFT
    np.save(f"{output_prefix}_power_spectrum.npy", power_spectrum)  # Zapisujemy widmo mocy
    logging.info(f"Pośrednie wyniki zapisane dla {output_prefix}")  # Informacja o zapisanych plikach

def generate_plots(autocorr, power_spectrum, output_prefix, fs):
    """
    Generuje wykresy funkcji autokorelacyjnych oraz widma mocy.
    Tworzy zarówno statyczne pliki PNG, jak i interaktywne wykresy w HTML.
    """
    # Wykres funkcji autokorelacyjnej (statyczny PNG)
    plt.figure(figsize=(10, 6))
    plt.plot(autocorr, label='Autokorelacja')
    plt.title('Funkcja Autokorelacyjna')
    plt.xlabel('Lag')  # Oś X to opóźnienie
    plt.ylabel('Wartość')  # Oś Y to wartość funkcji
    plt.legend()
    plt.grid(True)
    autocorr_png = f"{output_prefix}_autocorr.png"
    plt.savefig(autocorr_png)  # Zapisujemy wykres do pliku PNG
    plt.close()
    logging.info(f"Wykres funkcji autokorelacyjnej zapisany jako {autocorr_png}")

    # Wykres widma mocy (statyczny PNG)
    plt.figure(figsize=(10, 6))
    plt.plot(power_spectrum, label='Widmo Mocy')
    plt.title('Widmo Mocy')
    plt.xlabel('Częstotliwość (Hz)')  # Oś X to częstotliwość
    plt.ylabel('Moc (dB)')  # Oś Y to moc w dB
    plt.legend()
    plt.grid(True)
    fft_png = f"{output_prefix}.png"
    plt.savefig(fft_png)  # Zapisujemy widmo mocy do pliku PNG
    plt.close()
    logging.info(f"Wykres widma mocy zapisany jako {fft_png}")

    # Wykres interaktywny widma mocy (HTML)
    trace = go.Scatter(
        x=np.fft.fftfreq(len(power_spectrum), d=1/fs),  # Częstotliwości uzyskane z FFT
        y=power_spectrum,  # Widmo mocy
        mode='lines',
        name='Widmo Mocy'
    )
    layout = go.Layout(
        title='Interaktywne Widmo Mocy',
        xaxis=dict(title='Częstotliwość (Hz)'),  # Opis osi X
        yaxis=dict(title='Moc (dB)')  # Opis osi Y
    )
    fig = go.Figure(data=[trace], layout=layout)
    interactive_html = f"{output_prefix}_interactive.html"
    fig.write_html(interactive_html)  # Zapis interaktywnego wykresu jako HTML
    logging.info(f"Interaktywny wykres widma mocy zapisany jako {interactive_html}")

    return autocorr_png, fft_png, interactive_html  # Zwracamy ścieżki do zapisanych plików

def generate_report(results, output_dir):
    """
    Tworzy raport HTML, który zawiera podsumowanie analizy:
    dla każdego pliku generuje sekcję z wykresami autokorelacji i widm mocy.
    Dodatkowo zapisuje szczegółowe opisy wykresów.
    """
    report_path = os.path.join(output_dir, 'raport.html')
    try:
        with open(report_path, 'w') as report_file:
            report_file.write(f"<html><head><title>Raport Spectrum Analysis EP</title></head><body>\n")
            report_file.write(f"<h1>Raport Spectrum Analysis EP</h1>\n")
            report_file.write(f"<p>Data generacji raportu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            for result in results:
                report_file.write(f"<div class='file-section'>\n")
                report_file.write(f"    <h2>Plik danych: {result['file_name']}</h2>\n")
                for i in range(4):
                    report_file.write(f"    <h3>Funkcja autokorelacyjna {i+1} z pliku {result['file_name']}</h3>\n")
                    report_file.write(f"    <p><strong>Funkcja autokorelacyjna {i+1}:</strong></p>\n")
                    report_file.write(f"    <img src='{result['plots'][i]['autocorr_png']}' alt='Autokorelacja {i+1}' width='800'>\n")
                    report_file.write(f"    <p><strong>Widmo mocy {i+1}:</strong></p>\n")
                    report_file.write(f"    <img src='{result['plots'][i]['fft_png']}' alt='Widmo mocy {i+1}' width='800'>\n")
                    report_file.write(f"    <p><strong>Interaktywny wykres widma mocy {i+1}:</strong> <a href='{result['plots'][i]['interactive_html']}' target='_blank'>Otwórz</a></p>\n")
                    report_file.write(f"    <p>Widmo mocy opisuje rozkład mocy sygnału w różnych częstotliwościach. Wyższe wartości dB oznaczają większą moc dla danej częstotliwości.</p>\n")
                    report_file.write(f"    <p>Funkcja autokorelacyjna opisuje zależności czasowe sygnału, czyli jak wartości sygnału w różnych momentach czasu są ze sobą powiązane.</p>\n")
                    report_file.write(f"    <hr>\n")
                report_file.write(f"</div>\n")
            report_file.write(f"</body></html>")
        logging.info(f"Raport HTML został zapisany jako {report_path}")
    except Exception as e:
        logging.error(f"Błąd podczas generowania raportu HTML: {e}")

def main():
    setup_logging()  # Konfiguracja systemu logowania
    args = parse_arguments()  # Parsowanie argumentów linii poleceń

    logging.info("Rozpoczynanie analizy Spectrum Analysis EP")
    logging.info(f"Katalog z danymi wejściowymi: {args.data_dir}")
    logging.info(f"Plik wykluczeń: {args.exclude_file}")
    logging.info(f"Katalog z wynikami: {args.output_dir}")
    logging.info(f"Częstotliwość próbkowania: {args.fs} Hz")

    # Utwórz folder na wyniki, jeśli jeszcze nie istnieje
    os.makedirs(args.output_dir, exist_ok=True)

    # Wczytaj listę plików do wykluczenia z analizy
    exclude_files = load_exclude_list(args.exclude_file)

    # Pobierz listę plików do analizy, pomijając wykluczone
    data_files = get_data_files(args.data_dir, exclude_files)

    # Lista przechowująca wyniki analizy
    results = []

    # Przeanalizuj każdy plik z danych
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        logging.info(f"Przetwarzanie pliku: {file_name}")
        autocorr_functions = read_autocorrelation(file_path)  # Wczytanie funkcji autokorelacyjnych
        if not autocorr_functions:
            logging.warning(f"Pominięto plik {file_name} z powodu błędu podczas wczytywania.")
            continue  # Przeskocz ten plik, jeśli wystąpił błąd

        # Przechowujemy wykresy dla każdej z funkcji autokorelacyjnych
        plots_info = []
        for idx, autocorr in enumerate(autocorr_functions):
            # Symetryzacja funkcji
            sym_autocorr = symmetrize(autocorr)
            # Usunięcie DC offset
            cleaned_autocorr = remove_dc_offset(sym_autocorr)
            # Obliczenie FFT
            fft_result = compute_fft(cleaned_autocorr)
            # Obliczenie widma mocy
            power_spectrum = compute_power_spectrum(fft_result)

            # Zapisanie pośrednich wyników obliczeń
            output_prefix = os.path.join(args.output_dir, f"{os.path.splitext(file_name)[0]}_pcal{idx+1}")
            save_intermediate_results(output_prefix, autocorr, sym_autocorr, cleaned_autocorr, fft_result, power_spectrum)

            # Generowanie wykresów
            autocorr_png, fft_png, interactive_html = generate_plots(cleaned_autocorr, power_spectrum, output_prefix, args.fs)
            plots_info.append({
                'autocorr_png': os.path.basename(autocorr_png),
                'fft_png': os.path.basename(fft_png),
                'interactive_html': os.path.basename(interactive_html)
            })
        results.append({
            'file_name': file_name,
            'plots': plots_info
        })

    # Generowanie raportu HTML podsumowującego analizę
    generate_report(results, args.output_dir)
    logging.info("Analiza zakończona.")

if __name__ == "__main__":
    main()  # Wywołanie głównej funkcji programu
