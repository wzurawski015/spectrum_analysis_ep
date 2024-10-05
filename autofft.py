import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from scipy.fft import fft
from datetime import datetime

# Stała: częstotliwość próbkowania w Hz
FS = 1000

# Ścieżki do katalogów
DATA_DIR = 'data'
EXCLUDE_FILE = os.path.join(DATA_DIR, 'exclude')
OUTPUT_DIR = 'output'

def load_exclude_list(exclude_path):
    """
    Wczytuje listę plików do wykluczenia z pliku exclude.
    Jeśli plik nie istnieje, zwraca pustą listę.
    """
    exclude_files = []
    if os.path.exists(exclude_path):
        with open(exclude_path, 'r') as f:
            exclude_files = [line.strip() for line in f if line.strip()]
        print(f"Załadowano {len(exclude_files)} plików do wykluczenia.")
    else:
        print("Plik 'exclude' nie istnieje. Analizowane będą wszystkie pliki w katalogu 'data'.")
    return exclude_files

def get_data_files(data_dir, exclude_files):
    """
    Zwraca listę plików do analizy, pomijając te na liście wykluczeń.
    """
    all_files = glob.glob(os.path.join(data_dir, '*'))
    data_files = [f for f in all_files if os.path.isfile(f) and os.path.basename(f) not in exclude_files]
    print(f"Znaleziono {len(data_files)} plików do analizy.")
    return data_files

def read_autocorrelation(file_path):
    """
    Wczytuje funkcje autokorelacyjne z pliku danych.
    """
    try:
        data = np.loadtxt(file_path)
        channels = data[:, 0].astype(int)
        values = data[:, 1].astype(float)
        # Zakładamy, że każda funkcja autokorelacyjna jest podzielona na 4 części
        # Funkcje 1-3 mają 4097 próbek, funkcja 4 ma 4096 próbek
        autocorr1 = values[0:4097]
        autocorr2 = values[4097:8194]
        autocorr3 = values[8194:12291]
        autocorr4 = values[12291:16387]
        return [autocorr1, autocorr2, autocorr3, autocorr4]
    except Exception as e:
        print(f"Błąd podczas wczytywania pliku {file_path}: {e}")
        return []

def symmetrize(autocorr):
    """
    Symetryzuje funkcję autokorelacyjną, aby uzyskać rzeczywiste widmo mocy.
    """
    return np.concatenate((autocorr, autocorr[::-1]))

def remove_dc_offset(autocorr):
    """
    Usuwa składową stałą (DC offset) z funkcji autokorelacyjnej.
    """
    return autocorr - np.mean(autocorr)

def compute_fft(autocorr):
    """
    Wykonuje szybką transformatę Fouriera (FFT) na funkcji autokorelacyjnej.
    """
    fft_result = fft(autocorr)
    return np.abs(fft_result)

def compute_power_spectrum(fft_result):
    """
    Oblicza widmo mocy i przekształca je na skalę decybelową (dB).
    """
    power_spectrum = 10 * np.log10(fft_result + 1e-12)  # Dodanie 1e-12 dla stabilności logarytmu
    return power_spectrum

def generate_plots(autocorr, power_spectrum, output_prefix):
    """
    Generuje wykresy funkcji autokorelacyjnych i widm mocy.
    Tworzy zarówno wykresy statyczne (.png), jak i interaktywne (.html).
    """
    # Wykres funkcji autokorelacyjnej
    plt.figure(figsize=(10, 6))
    plt.plot(autocorr, label='Autokorelacja')
    plt.title('Funkcja Autokorelacyjna')
    plt.xlabel('Lag')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    autocorr_png = f"{output_prefix}_autocorr.png"
    plt.savefig(autocorr_png)
    plt.close()

    # Wykres widma mocy
    plt.figure(figsize=(10, 6))
    plt.plot(power_spectrum, label='Widmo Mocy')
    plt.title('Widmo Mocy')
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Moc (dB)')
    plt.legend()
    plt.grid(True)
    fft_png = f"{output_prefix}.png"
    plt.savefig(fft_png)
    plt.close()

    # Wykres interaktywny widma mocy
    trace = go.Scatter(x=np.fft.fftfreq(len(power_spectrum), d=1/FS), y=power_spectrum, mode='lines', name='Widmo Mocy')
    layout = go.Layout(title='Interaktywne Widmo Mocy',
                       xaxis=dict(title='Częstotliwość (Hz)'),
                       yaxis=dict(title='Moc (dB)'))
    fig = go.Figure(data=[trace], layout=layout)
    interactive_html = f"{output_prefix}_interactive.html"
    fig.write_html(interactive_html)

    return autocorr_png, fft_png, interactive_html

def generate_report(results, output_dir):
    """
    Tworzy raport HTML zawierający wszystkie wyniki analizy.
    """
    report_path = os.path.join(output_dir, 'raport.html')
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
                report_file.write(f"    <hr>\n")
            report_file.write(f"</div>\n")
        report_file.write(f"</body></html>")
    print(f"Raport HTML został zapisany jako {report_path}")

def main():
    # Utwórz folder output, jeśli nie istnieje
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Wczytaj listę plików do wykluczenia
    exclude_files = load_exclude_list(EXCLUDE_FILE)

    # Pobierz listę plików do analizy
    data_files = get_data_files(DATA_DIR, exclude_files)

    # Lista do przechowywania wyników
    results = []

    for file_path in data_files:
        file_name = os.path.basename(file_path)
        print(f"Przetwarzanie pliku: {file_name}")
        autocorr_functions = read_autocorrelation(file_path)
        if not autocorr_functions:
            continue
        plots_info = []
        for idx, autocorr in enumerate(autocorr_functions):
            # Symetryzacja
            sym_autocorr = symmetrize(autocorr)
            # Usunięcie DC offset
            cleaned_autocorr = remove_dc_offset(sym_autocorr)
            # FFT
            fft_result = compute_fft(cleaned_autocorr)
            # Widmo mocy
            power_spectrum = compute_power_spectrum(fft_result)
            # Generowanie wykresów
            output_prefix = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file_name)[0]}_pcal{idx+1}")
            autocorr_png, fft_png, interactive_html = generate_plots(cleaned_autocorr, power_spectrum, output_prefix)
            plots_info.append({
                'autocorr_png': os.path.basename(autocorr_png),
                'fft_png': os.path.basename(fft_png),
                'interactive_html': os.path.basename(interactive_html)
            })
        results.append({
            'file_name': file_name,
            'plots': plots_info
        })

    # Generowanie raportu HTML
    generate_report(results, OUTPUT_DIR)
    print("Analiza zakończona.")

if __name__ == "__main__":
    main()
