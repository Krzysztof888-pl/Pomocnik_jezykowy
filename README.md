# O aplikacji
**Aplikacja "Pomocnik Językowy" to zaawansowane narzędzie do wsparcia pracy nad tekstem, które łączy różne technologie sztucznej inteligencji, aby zwiększyć produktywność i kreatywność użytkowników. Udostępniona aplikacja umożliwia szybkie przetwarzanie i poprawę tekstów za pomocą modelu GPT-4o oraz szereg innych funkcji, które sprawiają, że praca z tekstami jest zdecydowanie szybsza, bardziej efektywna i wszechstronna.**

# Funkcjonalności
- **Nagrywanie notatek audio**: Możliwość nagrywania i zapisywania notatek w formie audio z zastosowaniem technologii OpenAI "whisper-1" do ich transkrypcji na tekst.

- **Transkrypcja audio na tekst**: Zautomatyzowane przekształcanie notatek audio na tekst.

- **Poprawa tekstu**: Wykorzystanie modelu GPT-4o do korekty błędów gramatycznych i stylistycznych oraz innych poprawek tekstu, w tym możliwość dokonania własnych zmian.

- **Tłumaczenie**: Szybkie tłumaczenie tekstu na różne języki z możliwością generowania wersji audio przy użyciu OpenAI "tts-1".

- **Generowanie mowy**: Synteza mowy z tekstu do audio, co pozwala na odsłuchanie notatek i ich łatwe udostępnianie.

- **Semantyczne wyszukiwanie notatek**: Zapisywanie i semantyczne wyszukiwanie notatek przy pomocy bazy danych wektorowych Qdrant oraz prowadzenie rozmów z użyciem ChatGPT-4o.

- Interfejs użytkownika: Aplikacja zbudowana na Pythonie z wykorzystaniem **Streamlit**, co zapewnia przyjazny i intuicyjny interfejs.

# Technologie
- OpenAI "**whisper-1**": Silnik transkrypcji audio.
- OpenAI "**GPT-4o**": Model do poprawy i tłumaczenia tekstu oraz interakcji z użytkownikiem.
- OpenAI "**text-embedding-3-large**": Tworzenie osadzeń tekstu w bazie danych.
- OpenAI "**tts-1**": Syntezator mowy do generowania wersji audio tekstu.
- **Qdrant**: Baza danych wektorowych do semantycznego przechowywania i przeszukiwania notatek.
- **Python** - użyty język programowania.
- **Streamlit**: Framework do budowy interfejsu użytkownika.
