from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder # type: ignore
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams


env = dotenv_values(".env")

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

QDRANT_COLLECTION_NAME = "notes"

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )

    return transcript.text

#
# DB
#
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(path=":memory:")
        
def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")

def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector=get_embedding(text=note_text),
                payload={
                    "text": note_text,
                },
            )
        ],
    )

def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=12)[0]
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": None,
            })

        return result
    
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })

        return result

#
# MAIN
#
st.set_page_config(page_title="🌍 Pomocnik Językowy", layout="wide")

with st.sidebar:
    st.sidebar.title("Panel boczny")

# OpenAi API key protection
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

# ChatGPT-4o session state initialization
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chat_text_source" not in st.session_state:
    st.session_state.chat_text_source = "note_text"

def toggle_chat():
    st.session_state.chat_active = not st.session_state.chat_active

st.sidebar.button(
    "Aktywuj/Dezaktywuj ChatGPT-4o",
    on_click=toggle_chat
)

# Lista dostępnych źródeł tekstu do rozmowy
text_sources = {
    "Pierwsza wersja notatki": "note_text",
    "Druga wersja notatki (poprawiona)": "note_text_corrected",
    "Tłumaczenie na BR ENG": "translated_text_br",
    "Tłumaczenie na US ENG": "translated_text_us",
    "Tłumaczenie na PL": "translated_text_pl",
    "Tłumaczenie na wybrany język": "translated_text_any",
    # Analogicznie możesz dodać źródła z search_tab, np.:
    "Pierwsza wersja wyszukanej notatki": "search_note_text",
    "Druga wersja wyszukanej notatki (poprawiona)": "search_note_text_corrected",
    "Tłumaczenie wyszukanej na BR ENG": "search_translated_text_br",
    "Tłumaczenie wyszukanej na US ENG": "search_translated_text_us",
    "Tłumaczenie wyszukanej na PL": "search_translated_text_pl",
    "Tłumaczenie wyszukanej na wybrany język": "search_translated_text_any",
}

if st.session_state.chat_active:
    st.sidebar.markdown("**Wybierz tekst do rozmowy z ChatGPT-4o:**")
    st.session_state.chat_text_source = st.sidebar.selectbox(
        "Źródło tekstu:",
        list(text_sources.keys()),
        key="chat_text_source_select"
    )
st.sidebar.markdown("**Informacje o aplikacji:**")
with st.sidebar.expander("O aplikacji i jej zastosowaniu"):
        st.markdown('''
                    Aplikacja "🌍 Pomocnik Językowy" została stworzona przede wszystkim po to, aby wspierać pracę nad tekstem, bardzo przyspieszając ten proces oraz poprawiając błędy, w tym gramatyczne i stylistyczne. Możemy tutaj dowolnie usprawniać tekst i przetwarzać go, zaczynając od nagrywania notatek audio. To znacznie przyspiesza pracę, ponieważ następnie możemy te nagrania transkrybować na tekst, a później poprawiać go za pomocą czatu GPT. Proces ten jest bardzo szybki. Dodatkowo możemy tłumaczyć ten tekst na różne języki. Jeśli to nie wystarcza, możemy nawet odsłuchać przetworzone notatki, generując audio w wybranych językach, a nawet wykorzystać je jako załącznik do wiadomości e-mail lub w różnego rodzaju komunikatorach. To nie wszystko; zapisane notatki możemy wyszukiwać semantycznie, co jest bardzo pomocne, gdy mamy ich sto lub więcej, a nie zawsze pamiętamy jakiej dokładnie frazy użyć do wyszukiwania - możemy użyć innych słów opisujących to czego szukamy. Na koniec, z nowymi lub odnalezionymi notatkami możemy pracować z ChatGPT-4o, przetwarzając je dowolnie. Połączenie tych wszystkich funkcji w jednej przestrzeni aplikacji znacznie przyspiesza pracę nad różnorodnym tekstem, od e-maili w różnych językach i stylach po różne formy wyrazu, takie jak wpisy na social media itp. - to jest potężne narzędzie do pracy nad tekstem
                    ''')

with st.sidebar.expander("Informacje o funkcjonalności i założeniach zastosowanych w aplikacji"):
        st.markdown('''
        Aplikacja "🌍 Pomocnik Językowy" umożliwia nagrywanie notatek audio, ich transkrypcję na tekst, poprawę tekstu przez model GPT-4o oraz tłumaczenie na różne języki wraz z utworzeniem wersji audio tych tłumaczeń, aby m.in. móc załączyć je do wiadomości e-mail lub innych komunikatorów. Notatki można również wyszukiwać semantycznie w bazie danych, a także rozmawiać o nich z ChatGPT-4o w różnych językach, dowolnie przekształcając ich treść i styl w zależności od potrzeb – sky is the limit
        ### Główne funkcjonalności:
        - Nagrywanie notatek audio lub wprowadzanie tekstu ręcznie/wklejenie
        - Transkrypcja audio na tekst
        - Poprawa tekstu przez model GPT-4o
        - Tłumaczenie na różne języki
        - Generowanie mowy z tekstu do audio
        - Rozmowa z modelem GPT-4o w odniesieniu do notatek
        - Zapisywanie notatek w wektorowej bazie danych Qdrant    
        - Semantyczne wyszukiwanie notatek w bazie danych Qdrant         
        ''')
with st.sidebar.expander("Zastosowane technologie i modele AI"):
        st.markdown('''
                    ### Zastosowane Technologie i modele AI:
        - OpenAI "**whisper-1**" do transkrypcji audio            
        - OpenAI "**GPT-4o**" do poprawy tekstu, tłumaczenia i dalszej dowolnej obróbki tekstu z ChatGPT-4o
        - OpenAI "**text-embedding-3-large**" do tworzenia osadzeń tekstu
        - OpenAI "**tts-1**" do syntezy mowy
        - **Qdrant** jako baza danych wektorowych do przechowywania i semantycznego wyszukiwania notatek
        - **Python** i jego biblioteka **Streamlit** jako framework do budowy interfejsu użytkownika
        - i najważniejsza technologia umysłu autora umożliwiająca łączenie złożonych światów ludzkich idei i myśli z potężnymi możliwościami nowoczesnej sztucznej inteligencji - zmierzamy w kierunku tworzenia coraz większych efektów synergii między ludźmi a AI, celem zwiększenia produktywności i kreatywności w tempie wykładniczym
        ''')
with st.sidebar.expander("Przydatne podpowiedzi dla początkujących użytkowników"):
        st.markdown('''
                    - Wszystkie edytowalne pola tekstowe można rozszerzać, przeciągając ich dolny prawy róg - w ten sposób można wygodniej edytować i porównywać dłuższe teksty
                    - Pole paska bocznego można poszerzać lub zmniejszać, przeciągając jego prawą krawędź - w ten sposób można wygodniej wybierać opcje i czytać informacje
                    - Pole paska bocznego można też schować lub pokazywać, klikając ikonę strzałki w lewym górnym rogu aplikacji - w ten sposób można zwiększyć przestrzeń roboczą głównego obszaru aplikacji
                    - Po transkrypcji audio na tekst, tekst można dalej edytować ręcznie, a następnie zapisać jego aktualną formę
                    ''')

with st.sidebar.expander("Ograniczenia aplikacji"):
        st.markdown('''
                    - Obecna forma aplikacji pomaga zademonstrować potencjał współpracy z AI oraz umożliwia prace ad hoc, ale nie służy do stałego przechowywania danych. Mimo to, aplikacja na tym etapie pozwala na pobieranie wygenerowanych plików audio oraz kopiowanie tekstów do plików tekstowych, takich jak txt czy docx. - warto ją traktować jako bazę wyjściową do konfiguracji rozwiązań uszytych na miarę potrzeb użytkownika
                    - Notatki są przechowywane tylko w pamięci (RAM) i znikają po zamknięciu aplikacji - (istnieje możliwość uzupełnienia tej funkcjonalności, ale to się oczywiście wiąże z kosztami przechowywania danych oraz skonfigurowaniem bazy danych na stałe wraz z jej zabezpieczeniem przed dostępem osób trzecich)
                    - Brak możliwości zapisywania notatek do pliku np. .txt (istnieje możliwość uzupełnienia tej funkcjonalności)
                    - Brak możliwości edytowania notatek po ich zapisaniu, ale można je wyszukiwać i ponownie przetwarzać, a następnie zapisać jej nową formę jako kolejną notatkę
                    - Aplikacja jest skierowana do polskiego użytkownika, ale AI myśli w uniwersalnym języku, więc można z nią już teraz rozmawiać/pracować w różnych językach. W przyszłości jej interfejs można łatwo przetłumaczyć na inny język - ba, sama umie to zrobić (przetłumaczyć dobrze tekst).
                    ''')
with st.sidebar.expander("Kontakt z autorem aplikacji"):
        st.markdown('''
                    - Zapraszam do współpracy i kontaktu poprzez LinkedIn: www.linkedin.com/in/krzysztof-bożek-59830b95
                     
                    ''')

st.title("Notatki Audio / Tekst do opracowania")

if st.session_state.chat_active:
    col_main, col_chat = st.columns([1, 1])
    with col_main:


        assure_db_collection_exists()
        add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

        with add_tab:
            # --- Sekcja nagrywania i transkrypcji ---
            note_audio = audiorecorder(
                start_prompt="Nagraj notatkę",
                stop_prompt="Zatrzymaj nagrywanie",
            )
            if note_audio:
                audio = BytesIO()
                note_audio.export(audio, format="mp3")
                st.session_state["note_audio_bytes"] = audio.getvalue()
                current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
                if st.session_state["note_audio_bytes_md5"] != current_md5:
                    st.session_state["note_audio_bytes_md5"] = current_md5

                st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

                if st.button("Transkrybuj audio"):
                    transcribed = transcribe_audio(st.session_state["note_audio_bytes"])
                    # Ustaw transkrypcję jako tekst do edycji
                    st.session_state["note_text"] = transcribed

            # --- Pole do edycji notatki (zawsze widoczne) ---
            st.session_state["note_text"] = st.text_area(
                "**Pierwsza wersja notatki:** Edytuj notatkę (możesz wpisać, wkleić lub użyć transkrypcji audio, a następnie dalej edytować):",
                value=st.session_state.get("note_text", ""),
                key="note_text_area"
            )

            # --- Sekcja poprawy przez GPT-4o ---
            if st.button("Popraw notatkę przez ChatGPT-4o"):
                openai_client = get_openai_client()
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Jesteś pomocnym asystentem, który wykrywa język przetwarzanego tekstu, a następnie koncentruje się jedynie na poprawie ewentualnych błędów w tym tekście, w tym samym języku: np. jeśli wykryjesz tekst napisany po polsku, popraw zgodnie z zasadami języka polskiego; jeśli wykryjesz tekst napisany po angielsku, popraw zgodnie z zasadami języka angielskiego. Poprawiaj tekst pod względem gramatycznym, stylistycznym, składniowym i ortograficznym. Popraw tylko błędy, nie zmieniaj sensu wypowiedzi."},
                        {"role": "user", "content": st.session_state["note_text"]},
                    ],
                    max_tokens=1024,
                )
                st.session_state["note_text_corrected"] = response.choices[0].message.content

            # Wyświetl poprawioną wersję, jeśli istnieje
            note_to_save = st.session_state.get("note_text", "")
            if "note_text_corrected" in st.session_state:
                st.session_state["note_text_corrected"] = st.text_area(
                    "**Druga wersja notatki:** Poprawiona notatka (możesz edytować):",
                    value=st.session_state["note_text_corrected"],
                    key="note_text_corrected_area"
                )
                note_to_save = st.session_state["note_text_corrected"]

            # --- Zapis notatki ---
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.get("note_text", "") and st.button("Zapisz pierwszą wersję notatki"):
                    add_note_to_db(note_text=st.session_state["note_text"])
                    st.toast("Pierwsza wersja notatki zapisana", icon="✅")
            with col2:
                if "note_text_corrected" in st.session_state and st.session_state["note_text_corrected"]:
                    if st.button("Zapisz drugą wersję notatki poprawioną przez GPT-4o"):
                        add_note_to_db(note_text=st.session_state["note_text_corrected"])
                        st.toast("Druga wersja notatki zapisana", icon="✅")
                        # Wyczyść poprawioną notatkę po zapisie (opcjonalnie)
                        # del st.session_state["note_text_corrected"]

            ##### UWAGA -PONIŻEJ KOD, który trzeba wciąć pod add_tab
            # Tworzenie czterech zakładek
            tab1, tab2, tab3, tab4 = st.tabs([
                "🇬🇧 Tłumaczenie na British English", 
                "🇺🇸 Tłumaczenie na American English", 
                "🇵🇱 Tłumaczenie na Polski", 
                "🌍 Tłumaczenie na wybrany język"
            ])

            with tab1:
                st.header("🇬🇧 Tłumaczenie na British English")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                # Wybór wersji notatki
                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select"
                    )

                    # Pobierz wybrany tekst
                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    # Przycisk tłumaczenia
                    if st.button("Przetłumacz na brytyjski angielski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na brytyjski angielski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_br"] = response.choices[0].message.content

                    # Pole do edycji tłumaczenia
                    if "translated_text_br" in st.session_state:
                        st.session_state["translated_text_br"] = st.text_area(
                            "Tłumaczenie na BR ENG (możesz edytować):",
                            value=st.session_state["translated_text_br"],
                            key="translated_text_br_area"
                        )

                        # Wybór głosu do syntezy mowy
                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select"
                        )

                        # Przycisk do generowania audio z tłumaczenia
                        if st.button("Wygeneruj audio z tłumaczenia na British English"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice,
                                input=st.session_state["translated_text_br"]
                            )
                            st.session_state["tts_br_audio"] = tts_response.content

                        # Odtwarzacz audio, jeśli audio zostało wygenerowane
                        if "tts_br_audio" in st.session_state:
                            st.audio(st.session_state["tts_br_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

            with tab2:
                st.header("🇺🇸 Tłumaczenie na American English")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_us"
                    )

                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    if st.button("Przetłumacz na amerykański angielski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na amerykański angielski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_us"] = response.choices[0].message.content

                    if "translated_text_us" in st.session_state:
                        st.session_state["translated_text_us"] = st.text_area(
                            "Tłumaczenie na US ENG (możesz edytować):",
                            value=st.session_state["translated_text_us"],
                            key="translated_text_us_area"
                        )

                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_us = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_us"
                        )

                        if st.button("Wygeneruj audio z tłumaczenia na American English"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_us,
                                input=st.session_state["translated_text_us"]
                            )
                            st.session_state["tts_us_audio"] = tts_response.content

                        if "tts_us_audio" in st.session_state:
                            st.audio(st.session_state["tts_us_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

            with tab3:
                st.header("🇵🇱 Tłumaczenie na Polski")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_pl"
                    )

                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    if st.button("Przetłumacz na polski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na język polski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_pl"] = response.choices[0].message.content

                    if "translated_text_pl" in st.session_state:
                        st.session_state["translated_text_pl"] = st.text_area(
                            "Tłumaczenie na polski (możesz edytować):",
                            value=st.session_state["translated_text_pl"],
                            key="translated_text_pl_area"
                        )

                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_pl = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_pl"
                        )

                        if st.button("Wygeneruj audio z tłumaczenia na polski"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_pl,
                                input=st.session_state["translated_text_pl"]
                            )
                            st.session_state["tts_pl_audio"] = tts_response.content

                        if "tts_pl_audio" in st.session_state:
                            st.audio(st.session_state["tts_pl_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

            with tab4:
                st.header("🌍 Tłumaczenie na wybrany język")
                st.write("Wybierz wersję notatki oraz język docelowy do tłumaczenia:")

                # Wersje notatki do wyboru
                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                # Lista języków obsługiwanych przez TTS-1 (OpenAI, stan na czerwiec 2024)
                # Kod języka: (nazwa wyświetlana, prompt do GPT, voice_hint)
                tts_languages = {
                    "en": ("English", "angielski", "alloy"),  # alloy, onyx, echo, fable, nova, shimmer
                    "de": ("German", "niemiecki", "echo"),
                    "es": ("Spanish", "hiszpański", "fable"),
                    "fr": ("French", "francuski", "nova"),
                    "it": ("Italian", "włoski", "onyx"),
                    "pt": ("Portuguese", "portugalski", "shimmer"),
                    "pl": ("Polish", "polski", "alloy"),
                    "tr": ("Turkish", "turecki", "alloy"),
                    "hi": ("Hindi", "hindi", "alloy"),
                    "ja": ("Japanese", "japoński", "alloy"),
                    "ko": ("Korean", "koreański", "alloy"),
                    "zh": ("Chinese", "chiński uproszczony", "alloy"),
                    # Dodaj inne języki obsługiwane przez TTS-1 jeśli pojawią się w przyszłości
                }

                language_display = [f"{v[0]} ({k})" for k, v in tts_languages.items()]

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_any"
                    )

                    # Pobierz wybrany tekst
                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    # Wybór języka docelowego
                    selected_lang_display = st.selectbox(
                        "Wybierz język docelowy do tłumaczenia:",
                        language_display,
                        key="target_language_select"
                    )
                    # Wyciągnij kod języka i nazwę do promptu
                    selected_lang_code = selected_lang_display.split("(")[-1].replace(")", "").strip()
                    selected_lang_prompt = tts_languages[selected_lang_code][1]

                    # Przycisk tłumaczenia
                    if st.button("Przetłumacz na wybrany język"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": f"Jesteś tłumaczem. Przetłumacz poniższy tekst na {selected_lang_prompt}, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_any"] = response.choices[0].message.content
                        st.session_state["translated_lang_code"] = selected_lang_code

                    # Pole do edycji tłumaczenia
                    if "translated_text_any" in st.session_state:
                        st.session_state["translated_text_any"] = st.text_area(
                            f"Tłumaczenie na {tts_languages.get(st.session_state.get('translated_lang_code', 'en'), ('Wybrany język',))[0]} (możesz edytować):",
                            value=st.session_state["translated_text_any"],
                            key="translated_text_any_area"
                        )

                        # Wybór głosu do syntezy mowy
                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_any = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_any"
                        )

                        # Przycisk do generowania audio z tłumaczenia
                        if st.button("Wygeneruj audio z tłumaczenia na wybrany język"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_any,
                                input=st.session_state["translated_text_any"],
                                response_format="mp3",
                                speed=1.0,
                            )
                            st.session_state["tts_any_audio"] = tts_response.content

                        # Odtwarzacz audio, jeśli audio zostało wygenerowane
                        if "tts_any_audio" in st.session_state:
                            st.audio(st.session_state["tts_any_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")
                
        with search_tab:

            query = st.text_input("Wyszukaj notatkę")
            notes = []
            if st.button("Szukaj"):
                notes = list_notes_from_db(query)
                st.session_state["search_results"] = notes  # zapisz wyniki do session_state
            elif "search_results" in st.session_state:
                notes = st.session_state["search_results"]

            selected_note_idx = None
            if notes:
                st.subheader("Wyniki wyszukiwania:")
                note_labels = []
                for idx, note in enumerate(notes):
                    # Wyświetl pełną notatkę i pełny score
                    with st.container(border=True):
                        st.markdown(note["text"])
                        if note["score"] is not None:
                            st.markdown(f':violet[score: {note["score"]}]')
                    # Przygotuj skrócony opis do wyboru
                    text_short = note["text"][:60].replace("\n", " ") + ("..." if len(note["text"]) > 60 else "")
                    score = f" (score: {note['score']})" if note["score"] is not None else ""
                    note_labels.append(text_short + score)

                # Pozwól użytkownikowi wybrać notatkę do dalszych akcji
                selected_note_idx = st.radio(
                    "Wybierz notatkę do dalszych akcji:",
                    options=list(range(len(notes))),
                    format_func=lambda i: note_labels[i],
                    key="search_selected_note_idx"
                )

            # Jeśli wybrano notatkę, pokaż edycję i akcje jak w add_tab
            if selected_note_idx is not None:
                # --- Edycja wybranej notatki ---
                if "search_note_text" not in st.session_state or st.session_state.get("last_selected_note_idx") != selected_note_idx:
                    st.session_state["search_note_text"] = notes[selected_note_idx]["text"]
                    st.session_state["search_note_text_corrected"] = ""
                    st.session_state["last_selected_note_idx"] = selected_note_idx

                st.session_state["search_note_text"] = st.text_area(
                    "**Pierwsza wersja wyszukanej notatki:** Edytuj notatkę (możesz modyfikować przed dalszymi akcjami):",
                    value=st.session_state["search_note_text"],
                    key="search_note_text_area"
                )

                # --- Poprawa przez GPT-4o ---
                if st.button("Popraw notatkę przez ChatGPT-4o", key="search_correct_btn"):
                    openai_client = get_openai_client()
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Jesteś pomocnym asystentem, który wykrywa język przetwarzanego tekstu, a następnie koncentruje się jedynie na poprawie ewentualnych błędów w tym tekście, w tym samym języku: np. jeśli wykryjesz tekst napisany po polsku, popraw zgodnie z zasadami języka polskiego; jeśli wykryjesz tekst napisany po angielsku, popraw zgodnie z zasadami języka angielskiego. Poprawiaj tekst pod względem gramatycznym, stylistycznym, składniowym i ortograficznym. Popraw tylko błędy, nie zmieniaj sensu wypowiedzi."},
                            {"role": "user", "content": st.session_state["search_note_text"]},
                        ],
                        max_tokens=1024,
                    )
                    st.session_state["search_note_text_corrected"] = response.choices[0].message.content

                # --- Wyświetl poprawioną wersję, jeśli istnieje ---
                note_to_translate = st.session_state["search_note_text"]
                if st.session_state.get("search_note_text_corrected"):
                    st.session_state["search_note_text_corrected"] = st.text_area(
                        "**Druga wersja wyszukanej notatki:** Poprawiona notatka (możesz edytować):",
                        value=st.session_state["search_note_text_corrected"],
                        key="search_note_text_corrected_area"
                    )
                    note_to_translate = st.session_state["search_note_text_corrected"]

                st.markdown("---")
                st.subheader("Akcje dla wybranej notatki:")

                # --- Zakładki jak w add_tab, ale operujące na note_to_translate ---
                tab1s, tab2s, tab3s, tab4s = st.tabs([
                    "🇬🇧 Tłumaczenie na British English", 
                    "🇺🇸 Tłumaczenie na American English", 
                    "🇵🇱 Tłumaczenie na Polski", 
                    "🌍 Tłumaczenie na wybrany język"
                ])

                with tab1s:
                    st.header("🇬🇧 Tłumaczenie na British English")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    # Wybór wersji notatki
                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select"
                        )

                        # Pobierz wybrany tekst
                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        # Przycisk tłumaczenia
                        if st.button("Przetłumacz na brytyjski angielski", key="search_translate_br"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na brytyjski angielski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_br"] = response.choices[0].message.content

                        # Pole do edycji tłumaczenia
                        if "search_translated_text_br" in st.session_state:
                            st.session_state["search_translated_text_br"] = st.text_area(
                                "Tłumaczenie na BR ENG (możesz edytować):",
                                value=st.session_state["search_translated_text_br"],
                                key="search_translated_text_br_area"
                            )

                            # Wybór głosu do syntezy mowy
                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_br"
                            )

                            # Przycisk do generowania audio z tłumaczenia
                            if st.button("Wygeneruj audio z tłumaczenia na British English", key="search_tts_br"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice,
                                    input=st.session_state["search_translated_text_br"]
                                )
                                st.session_state["search_tts_br_audio"] = tts_response.content

                            # Odtwarzacz audio, jeśli audio zostało wygenerowane
                            if "search_tts_br_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_br_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

                with tab2s:
                    st.header("🇺🇸 Tłumaczenie na American English")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_us"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        if st.button("Przetłumacz na amerykański angielski", key="search_translate_us"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na amerykański angielski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_us"] = response.choices[0].message.content

                        if "search_translated_text_us" in st.session_state:
                            st.session_state["search_translated_text_us"] = st.text_area(
                                "Tłumaczenie na US ENG (możesz edytować):",
                                value=st.session_state["search_translated_text_us"],
                                key="search_translated_text_us_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_us = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_us"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na American English", key="search_tts_us"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_us,
                                    input=st.session_state["search_translated_text_us"]
                                )
                                st.session_state["search_tts_us_audio"] = tts_response.content

                            if "search_tts_us_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_us_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")

                with tab3s:
                    st.header("🇵🇱 Tłumaczenie na Polski")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_pl"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        if st.button("Przetłumacz na polski", key="search_translate_pl"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na język polski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_pl"] = response.choices[0].message.content

                        if "search_translated_text_pl" in st.session_state:
                            st.session_state["search_translated_text_pl"] = st.text_area(
                                "Tłumaczenie na polski (możesz edytować):",
                                value=st.session_state["search_translated_text_pl"],
                                key="search_translated_text_pl_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_pl = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_pl"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na polski", key="search_tts_pl"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_pl,
                                    input=st.session_state["search_translated_text_pl"]
                                )
                                st.session_state["search_tts_pl_audio"] = tts_response.content

                            if "search_tts_pl_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_pl_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")

                with tab4s:
                    st.header("🌍 Tłumaczenie na wybrany język")
                    st.write("Wybierz wersję wyszukanej notatki oraz język docelowy do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    # Lista języków obsługiwanych przez TTS-1 (OpenAI, stan na czerwiec 2024)
                    tts_languages = {
                        "en": ("English", "angielski", "alloy"),
                        "de": ("German", "niemiecki", "echo"),
                        "es": ("Spanish", "hiszpański", "fable"),
                        "fr": ("French", "francuski", "nova"),
                        "it": ("Italian", "włoski", "onyx"),
                        "pt": ("Portuguese", "portugalski", "shimmer"),
                        "pl": ("Polish", "polski", "alloy"),
                        "tr": ("Turkish", "turecki", "alloy"),
                        "hi": ("Hindi", "hindi", "alloy"),
                        "ja": ("Japanese", "japoński", "alloy"),
                        "ko": ("Korean", "koreański", "alloy"),
                        "zh": ("Chinese", "chiński uproszczony", "alloy"),
                    }
                    language_display = [f"{v[0]} ({k})" for k, v in tts_languages.items()]

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_any"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        selected_lang_display = st.selectbox(
                            "Wybierz język docelowy do tłumaczenia:",
                            language_display,
                            key="search_target_language_select"
                        )
                        selected_lang_code = selected_lang_display.split("(")[-1].replace(")", "").strip()
                        selected_lang_prompt = tts_languages[selected_lang_code][1]

                        if st.button("Przetłumacz na wybrany język", key="search_translate_any"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": f"Jesteś tłumaczem. Przetłumacz poniższy tekst na {selected_lang_prompt}, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_any"] = response.choices[0].message.content
                            st.session_state["search_translated_lang_code"] = selected_lang_code

                        if "search_translated_text_any" in st.session_state:
                            st.session_state["search_translated_text_any"] = st.text_area(
                                f"Tłumaczenie na {tts_languages.get(st.session_state.get('search_translated_lang_code', 'en'), ('Wybrany język',))[0]} (możesz edytować):",
                                value=st.session_state["search_translated_text_any"],
                                key="search_translated_text_any_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_any = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_any"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na wybrany język", key="search_tts_any"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_any,
                                    input=st.session_state["search_translated_text_any"],
                                    response_format="mp3",
                                    speed=1.0,
                                )
                                st.session_state["search_tts_any_audio"] = tts_response.content

                            if "search_tts_any_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_any_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")
    with col_chat:
        st.subheader("💬 Rozmowa z ChatGPT-4o o wybranym tekście")
        text_key = text_sources[st.session_state.chat_text_source]
        text_for_chat = st.session_state.get(text_key, "")

        # Jeśli historia jest pusta, ustaw pierwszy prompt z kontekstem
        if not st.session_state.chat_history and text_for_chat:
            st.session_state.chat_history.append({
                "role": "system",
                "content": f"Jesteś ekspertem językowym. Odpowiadasz na pytania dotyczące wybranego tekstu:\n\n{text_for_chat}"
            })

        # Zapamiętaj poprzednie źródło tekstu
        if "prev_chat_text_source" not in st.session_state:
            st.session_state["prev_chat_text_source"] = st.session_state.chat_text_source

        # Jeśli zmieniono źródło tekstu, zresetuj historię i ustaw nowy systemowy prompt
        if st.session_state["prev_chat_text_source"] != st.session_state.chat_text_source:
            st.session_state.chat_history = []
            text_key = text_sources[st.session_state.chat_text_source]
            text_for_chat = st.session_state.get(text_key, "")
            if text_for_chat:
                st.session_state.chat_history.append({
                    "role": "system",
                    "content": f"Jesteś ekspertem językowym. Odpowiadasz na pytania dotyczące wybranego tekstu:\n\n{text_for_chat}"
                })
            st.session_state["prev_chat_text_source"] = st.session_state.chat_text_source

        st.markdown("**Tekst do rozmowy:**")
        st.write(text_for_chat)

        if st.session_state.get("clear_chat_input"):
            st.session_state["chat_user_input"] = ""
            st.session_state["clear_chat_input"] = False

        user_input = st.text_input(
            "Zadaj pytanie dotyczące powyższego tekstu lub poproś o wyjaśnienie, np. 1. Czy tekst jest poprawny pod względem gramatycznym i stylistycznym? Wykonaj analizę. 2. Przekształć tekst na bardziej formalny styl - zaproponuj dwie wersje",
            value=st.session_state.get("chat_user_input", ""),
            key="chat_user_input",
            # placeholder="Czy tekst jest poprawny pod względem gramatycznym i stylistycznym? Wykonaj analizę"
        )

        if user_input:
            # --- AKTUALIZUJ SYSTEMOWY PROMPT Z NAJNOWSZYM TEKSTEM ---
            text_key = text_sources[st.session_state.chat_text_source]
            text_for_chat = st.session_state.get(text_key, "")
            # Usuń stary systemowy prompt (jeśli istnieje)
            if st.session_state.chat_history and st.session_state.chat_history[0]["role"] == "system":
                st.session_state.chat_history[0] = {
                    "role": "system",
                    "content": f"Jesteś ekspertem językowym. Odpowiadasz na pytania dotyczące wybranego tekstu:\n\n{text_for_chat}"
                }
            else:
                st.session_state.chat_history.insert(0, {
                    "role": "system",
                    "content": f"Jesteś ekspertem językowym. Odpowiadasz na pytania dotyczące wybranego tekstu:\n\n{text_for_chat}"
                })

            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                openai_client = get_openai_client()
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=st.session_state.chat_history,
                    max_tokens=1024
                )
                answer = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as e:
                answer = f"Błąd komunikacji z OpenAI: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            st.session_state["clear_chat_input"] = True

        # Wyświetl historię rozmowy
        # for msg in reversed(st.session_state.chat_history):
        for msg in st.session_state.chat_history:    
            if msg["role"] == "user":
                st.markdown(f"**Ty:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**ChatGPT-4o:** {msg['content']}")
            elif msg["role"] == "system":
                st.markdown(f"**Kontekst rozmowy:** {msg['content']}")

        if st.button("Wyczyść historię rozmowy"):
            st.session_state.chat_history = []
            st.session_state["clear_chat_input"] = True
            st.rerun()

        # if st.button("Wyczyść historię rozmowy"):
        #     st.session_state.chat_history = []
else:
        assure_db_collection_exists()
        add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

        with add_tab:
            # --- Sekcja nagrywania i transkrypcji ---
            note_audio = audiorecorder(
                start_prompt="Nagraj notatkę",
                stop_prompt="Zatrzymaj nagrywanie",
            )
            if note_audio:
                audio = BytesIO()
                note_audio.export(audio, format="mp3")
                st.session_state["note_audio_bytes"] = audio.getvalue()
                current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
                if st.session_state["note_audio_bytes_md5"] != current_md5:
                    st.session_state["note_audio_bytes_md5"] = current_md5

                st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

                if st.button("Transkrybuj audio"):
                    transcribed = transcribe_audio(st.session_state["note_audio_bytes"])
                    # Ustaw transkrypcję jako tekst do edycji
                    st.session_state["note_text"] = transcribed

            # --- Pole do edycji notatki (zawsze widoczne) ---
            st.session_state["note_text"] = st.text_area(
                "**Pierwsza wersja notatki:** Edytuj notatkę (możesz wpisać, wkleić lub użyć transkrypcji audio, a następnie dalej edytować):",
                value=st.session_state.get("note_text", ""),
                key="note_text_area"
            )

            # --- Sekcja poprawy przez GPT-4o ---
            if st.button("Popraw notatkę przez ChatGPT-4o"):
                openai_client = get_openai_client()
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Jesteś pomocnym asystentem, który wykrywa język przetwarzanego tekstu, a następnie koncentruje się jedynie na poprawie ewentualnych błędów w tym tekście, w tym samym języku: np. jeśli wykryjesz tekst napisany po polsku, popraw zgodnie z zasadami języka polskiego; jeśli wykryjesz tekst napisany po angielsku, popraw zgodnie z zasadami języka angielskiego. Poprawiaj tekst pod względem gramatycznym, stylistycznym, składniowym i ortograficznym. Popraw tylko błędy, nie zmieniaj sensu wypowiedzi."},
                        {"role": "user", "content": st.session_state["note_text"]},
                    ],
                    max_tokens=1024,
                )
                st.session_state["note_text_corrected"] = response.choices[0].message.content

            # Wyświetl poprawioną wersję, jeśli istnieje
            note_to_save = st.session_state.get("note_text", "")
            if "note_text_corrected" in st.session_state:
                st.session_state["note_text_corrected"] = st.text_area(
                    "**Druga wersja notatki:** Poprawiona notatka (możesz edytować):",
                    value=st.session_state["note_text_corrected"],
                    key="note_text_corrected_area"
                )
                note_to_save = st.session_state["note_text_corrected"]

            # --- Zapis notatki ---
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.get("note_text", "") and st.button("Zapisz pierwszą wersję notatki"):
                    add_note_to_db(note_text=st.session_state["note_text"])
                    st.toast("Pierwsza wersja notatki zapisana", icon="✅")
            with col2:
                if "note_text_corrected" in st.session_state and st.session_state["note_text_corrected"]:
                    if st.button("Zapisz drugą wersję notatki poprawioną przez GPT-4o"):
                        add_note_to_db(note_text=st.session_state["note_text_corrected"])
                        st.toast("Druga wersja notatki zapisana", icon="✅")
                        # Wyczyść poprawioną notatkę po zapisie (opcjonalnie)
                        # del st.session_state["note_text_corrected"]

            # Tworzenie czterech zakładek
            tab1, tab2, tab3, tab4 = st.tabs([
                "🇬🇧 Tłumaczenie na British English", 
                "🇺🇸 Tłumaczenie na American English", 
                "🇵🇱 Tłumaczenie na Polski", 
                "🌍 Tłumaczenie na wybrany język"
            ])

            with tab1:
                st.header("🇬🇧 Tłumaczenie na British English")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                # Wybór wersji notatki
                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select"
                    )

                    # Pobierz wybrany tekst
                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    # Przycisk tłumaczenia
                    if st.button("Przetłumacz na brytyjski angielski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na brytyjski angielski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_br"] = response.choices[0].message.content

                    # Pole do edycji tłumaczenia
                    if "translated_text_br" in st.session_state:
                        st.session_state["translated_text_br"] = st.text_area(
                            "Tłumaczenie na BR ENG (możesz edytować):",
                            value=st.session_state["translated_text_br"],
                            key="translated_text_br_area"
                        )

                        # Wybór głosu do syntezy mowy
                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select"
                        )

                        # Przycisk do generowania audio z tłumaczenia
                        if st.button("Wygeneruj audio z tłumaczenia na British English"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice,
                                input=st.session_state["translated_text_br"]
                            )
                            st.session_state["tts_br_audio"] = tts_response.content

                        # Odtwarzacz audio, jeśli audio zostało wygenerowane
                        if "tts_br_audio" in st.session_state:
                            st.audio(st.session_state["tts_br_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

            with tab2:
                st.header("🇺🇸 Tłumaczenie na American English")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_us"
                    )

                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    if st.button("Przetłumacz na amerykański angielski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na amerykański angielski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_us"] = response.choices[0].message.content

                    if "translated_text_us" in st.session_state:
                        st.session_state["translated_text_us"] = st.text_area(
                            "Tłumaczenie na US ENG (możesz edytować):",
                            value=st.session_state["translated_text_us"],
                            key="translated_text_us_area"
                        )

                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_us = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_us"
                        )

                        if st.button("Wygeneruj audio z tłumaczenia na American English"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_us,
                                input=st.session_state["translated_text_us"]
                            )
                            st.session_state["tts_us_audio"] = tts_response.content

                        if "tts_us_audio" in st.session_state:
                            st.audio(st.session_state["tts_us_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

            with tab3:
                st.header("🇵🇱 Tłumaczenie na Polski")
                st.write("Wybierz wersję notatki do tłumaczenia:")

                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_pl"
                    )

                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    if st.button("Przetłumacz na polski"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na język polski, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_pl"] = response.choices[0].message.content

                    if "translated_text_pl" in st.session_state:
                        st.session_state["translated_text_pl"] = st.text_area(
                            "Tłumaczenie na polski (możesz edytować):",
                            value=st.session_state["translated_text_pl"],
                            key="translated_text_pl_area"
                        )

                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_pl = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_pl"
                        )

                        if st.button("Wygeneruj audio z tłumaczenia na polski"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_pl,
                                input=st.session_state["translated_text_pl"]
                            )
                            st.session_state["tts_pl_audio"] = tts_response.content

                        if "tts_pl_audio" in st.session_state:
                            st.audio(st.session_state["tts_pl_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")


            with tab4:
                st.header("🌍 Tłumaczenie na wybrany język")
                st.write("Wybierz wersję notatki oraz język docelowy do tłumaczenia:")

                # Wersje notatki do wyboru
                note_options = []
                if st.session_state.get("note_text"):
                    note_options.append("Pierwsza wersja notatki")
                if st.session_state.get("note_text_corrected"):
                    note_options.append("Druga wersja notatki (poprawiona przez GPT-4o)")

                # Lista języków obsługiwanych przez TTS-1 (OpenAI, stan na czerwiec 2024)
                # Kod języka: (nazwa wyświetlana, prompt do GPT, voice_hint)
                tts_languages = {
                    "en": ("English", "angielski", "alloy"),  # alloy, onyx, echo, fable, nova, shimmer
                    "de": ("German", "niemiecki", "echo"),
                    "es": ("Spanish", "hiszpański", "fable"),
                    "fr": ("French", "francuski", "nova"),
                    "it": ("Italian", "włoski", "onyx"),
                    "pt": ("Portuguese", "portugalski", "shimmer"),
                    "pl": ("Polish", "polski", "alloy"),
                    "tr": ("Turkish", "turecki", "alloy"),
                    "hi": ("Hindi", "hindi", "alloy"),
                    "ja": ("Japanese", "japoński", "alloy"),
                    "ko": ("Korean", "koreański", "alloy"),
                    "zh": ("Chinese", "chiński uproszczony", "alloy"),
                    # Dodaj inne języki obsługiwane przez TTS-1 jeśli pojawią się w przyszłości
                }

                language_display = [f"{v[0]} ({k})" for k, v in tts_languages.items()]

                if note_options:
                    selected_note = st.radio(
                        "Wersja notatki:",
                        note_options,
                        key="translation_note_select_any"
                    )

                    # Pobierz wybrany tekst
                    if selected_note == "Pierwsza wersja notatki":
                        text_to_translate = st.session_state["note_text"]
                    else:
                        text_to_translate = st.session_state["note_text_corrected"]

                    # Wybór języka docelowego
                    selected_lang_display = st.selectbox(
                        "Wybierz język docelowy do tłumaczenia:",
                        language_display,
                        key="target_language_select"
                    )
                    # Wyciągnij kod języka i nazwę do promptu
                    selected_lang_code = selected_lang_display.split("(")[-1].replace(")", "").strip()
                    selected_lang_prompt = tts_languages[selected_lang_code][1]

                    # Przycisk tłumaczenia
                    if st.button("Przetłumacz na wybrany język"):
                        openai_client = get_openai_client()
                        response = openai_client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": f"Jesteś tłumaczem. Przetłumacz poniższy tekst na {selected_lang_prompt}, zachowując sens i styl oryginału."},
                                {"role": "user", "content": text_to_translate},
                            ],
                            max_tokens=1024,
                        )
                        st.session_state["translated_text_any"] = response.choices[0].message.content
                        st.session_state["translated_lang_code"] = selected_lang_code

                    # Pole do edycji tłumaczenia
                    if "translated_text_any" in st.session_state:
                        st.session_state["translated_text_any"] = st.text_area(
                            f"Tłumaczenie na {tts_languages.get(st.session_state.get('translated_lang_code', 'en'), ('Wybrany język',))[0]} (możesz edytować):",
                            value=st.session_state["translated_text_any"],
                            key="translated_text_any_area"
                        )

                        # Wybór głosu do syntezy mowy
                        voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                        selected_voice_any = st.selectbox(
                            "Wybierz typ/rodzaj głosu do syntezy mowy:",
                            options=voice_options,
                            key="tts_voice_select_any"
                        )

                        # Przycisk do generowania audio z tłumaczenia
                        if st.button("Wygeneruj audio z tłumaczenia na wybrany język"):
                            openai_client = get_openai_client()
                            tts_response = openai_client.audio.speech.create(
                                model="tts-1",
                                voice=selected_voice_any,
                                input=st.session_state["translated_text_any"],
                                response_format="mp3",
                                speed=1.0,
                            )
                            st.session_state["tts_any_audio"] = tts_response.content

                        # Odtwarzacz audio, jeśli audio zostało wygenerowane
                        if "tts_any_audio" in st.session_state:
                            st.audio(st.session_state["tts_any_audio"], format="audio/mp3")
                else:
                    st.info("Brak notatek do tłumaczenia. Dodaj lub popraw notatkę w zakładce 'Dodaj notatkę'.")
                
        with search_tab:

            query = st.text_input("Wyszukaj notatkę")
            notes = []
            if st.button("Szukaj"):
                notes = list_notes_from_db(query)
                st.session_state["search_results"] = notes  # zapisz wyniki do session_state
            elif "search_results" in st.session_state:
                notes = st.session_state["search_results"]

            selected_note_idx = None
            if notes:
                st.subheader("Wyniki wyszukiwania:")
                note_labels = []
                for idx, note in enumerate(notes):
                    # Wyświetl pełną notatkę i pełny score
                    with st.container(border=True):
                        st.markdown(note["text"])
                        if note["score"] is not None:
                            st.markdown(f':violet[score: {note["score"]}]')
                    # Przygotuj skrócony opis do wyboru
                    text_short = note["text"][:60].replace("\n", " ") + ("..." if len(note["text"]) > 60 else "")
                    score = f" (score: {note['score']})" if note["score"] is not None else ""
                    note_labels.append(text_short + score)

                # Pozwól użytkownikowi wybrać notatkę do dalszych akcji
                selected_note_idx = st.radio(
                    "Wybierz notatkę do dalszych akcji:",
                    options=list(range(len(notes))),
                    format_func=lambda i: note_labels[i],
                    key="search_selected_note_idx"
                )

            # Jeśli wybrano notatkę, pokaż edycję i akcje jak w add_tab
            if selected_note_idx is not None:
                # --- Edycja wybranej notatki ---
                if "search_note_text" not in st.session_state or st.session_state.get("last_selected_note_idx") != selected_note_idx:
                    st.session_state["search_note_text"] = notes[selected_note_idx]["text"]
                    st.session_state["search_note_text_corrected"] = ""
                    st.session_state["last_selected_note_idx"] = selected_note_idx

                st.session_state["search_note_text"] = st.text_area(
                    "**Pierwsza wersja wyszukanej notatki:** Edytuj notatkę (możesz modyfikować przed dalszymi akcjami):",
                    value=st.session_state["search_note_text"],
                    key="search_note_text_area"
                )

                # --- Poprawa przez GPT-4o ---
                if st.button("Popraw notatkę przez ChatGPT-4o", key="search_correct_btn"):
                    openai_client = get_openai_client()
                    response = openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "Jesteś pomocnym asystentem, który wykrywa język przetwarzanego tekstu, a następnie koncentruje się jedynie na poprawie ewentualnych błędów w tym tekście, w tym samym języku: np. jeśli wykryjesz tekst napisany po polsku, popraw zgodnie z zasadami języka polskiego; jeśli wykryjesz tekst napisany po angielsku, popraw zgodnie z zasadami języka angielskiego. Poprawiaj tekst pod względem gramatycznym, stylistycznym, składniowym i ortograficznym. Popraw tylko błędy, nie zmieniaj sensu wypowiedzi."},
                            {"role": "user", "content": st.session_state["search_note_text"]},
                        ],
                        max_tokens=1024,
                    )
                    st.session_state["search_note_text_corrected"] = response.choices[0].message.content

                # --- Wyświetl poprawioną wersję, jeśli istnieje ---
                note_to_translate = st.session_state["search_note_text"]
                if st.session_state.get("search_note_text_corrected"):
                    st.session_state["search_note_text_corrected"] = st.text_area(
                        "**Druga wersja wyszukanej notatki:** Poprawiona notatka (możesz edytować):",
                        value=st.session_state["search_note_text_corrected"],
                        key="search_note_text_corrected_area"
                    )
                    note_to_translate = st.session_state["search_note_text_corrected"]

                st.markdown("---")
                st.subheader("Akcje dla wybranej notatki:")

                # --- Zakładki jak w add_tab, ale operujące na note_to_translate ---
                tab1s, tab2s, tab3s, tab4s = st.tabs([
                    "🇬🇧 Tłumaczenie na British English", 
                    "🇺🇸 Tłumaczenie na American English", 
                    "🇵🇱 Tłumaczenie na Polski", 
                    "🌍 Tłumaczenie na wybrany język"
                ])

                with tab1s:
                    st.header("🇬🇧 Tłumaczenie na British English")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    # Wybór wersji notatki
                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select"
                        )

                        # Pobierz wybrany tekst
                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        # Przycisk tłumaczenia
                        if st.button("Przetłumacz na brytyjski angielski", key="search_translate_br"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na brytyjski angielski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_br"] = response.choices[0].message.content

                        # Pole do edycji tłumaczenia
                        if "search_translated_text_br" in st.session_state:
                            st.session_state["search_translated_text_br"] = st.text_area(
                                "Tłumaczenie na BR ENG (możesz edytować):",
                                value=st.session_state["search_translated_text_br"],
                                key="search_translated_text_br_area"
                            )

                            # Wybór głosu do syntezy mowy
                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_br"
                            )

                            # Przycisk do generowania audio z tłumaczenia
                            if st.button("Wygeneruj audio z tłumaczenia na British English", key="search_tts_br"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice,
                                    input=st.session_state["search_translated_text_br"]
                                )
                                st.session_state["search_tts_br_audio"] = tts_response.content

                            # Odtwarzacz audio, jeśli audio zostało wygenerowane
                            if "search_tts_br_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_br_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę w zakładce 'Dodaj notatkę'.")

                with tab2s:
                    st.header("🇺🇸 Tłumaczenie na American English")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_us"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        if st.button("Przetłumacz na amerykański angielski", key="search_translate_us"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na amerykański angielski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_us"] = response.choices[0].message.content

                        if "search_translated_text_us" in st.session_state:
                            st.session_state["search_translated_text_us"] = st.text_area(
                                "Tłumaczenie na US ENG (możesz edytować):",
                                value=st.session_state["search_translated_text_us"],
                                key="search_translated_text_us_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_us = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_us"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na American English", key="search_tts_us"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_us,
                                    input=st.session_state["search_translated_text_us"]
                                )
                                st.session_state["search_tts_us_audio"] = tts_response.content

                            if "search_tts_us_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_us_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")

                with tab3s:
                    st.header("🇵🇱 Tłumaczenie na Polski")
                    st.write("Wybierz wersję wyszukanej notatki do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_pl"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        if st.button("Przetłumacz na polski", key="search_translate_pl"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": "Jesteś tłumaczem. Przetłumacz poniższy tekst na język polski, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_pl"] = response.choices[0].message.content

                        if "search_translated_text_pl" in st.session_state:
                            st.session_state["search_translated_text_pl"] = st.text_area(
                                "Tłumaczenie na polski (możesz edytować):",
                                value=st.session_state["search_translated_text_pl"],
                                key="search_translated_text_pl_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_pl = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_pl"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na polski", key="search_tts_pl"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_pl,
                                    input=st.session_state["search_translated_text_pl"]
                                )
                                st.session_state["search_tts_pl_audio"] = tts_response.content

                            if "search_tts_pl_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_pl_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")

                with tab4s:
                    st.header("🌍 Tłumaczenie na wybrany język")
                    st.write("Wybierz wersję wyszukanej notatki oraz język docelowy do tłumaczenia:")

                    note_options = []
                    if st.session_state.get("search_note_text"):
                        note_options.append("Pierwsza wersja wyszukanej notatki")
                    if st.session_state.get("search_note_text_corrected"):
                        note_options.append("Druga wersja wyszukanej notatki (poprawiona przez GPT-4o)")

                    # Lista języków obsługiwanych przez TTS-1 (OpenAI, stan na czerwiec 2024)
                    tts_languages = {
                        "en": ("English", "angielski", "alloy"),
                        "de": ("German", "niemiecki", "echo"),
                        "es": ("Spanish", "hiszpański", "fable"),
                        "fr": ("French", "francuski", "nova"),
                        "it": ("Italian", "włoski", "onyx"),
                        "pt": ("Portuguese", "portugalski", "shimmer"),
                        "pl": ("Polish", "polski", "alloy"),
                        "tr": ("Turkish", "turecki", "alloy"),
                        "hi": ("Hindi", "hindi", "alloy"),
                        "ja": ("Japanese", "japoński", "alloy"),
                        "ko": ("Korean", "koreański", "alloy"),
                        "zh": ("Chinese", "chiński uproszczony", "alloy"),
                    }
                    language_display = [f"{v[0]} ({k})" for k, v in tts_languages.items()]

                    if note_options:
                        selected_note = st.radio(
                            "Wersja notatki:",
                            note_options,
                            key="search_translation_note_select_any"
                        )

                        if selected_note == "Pierwsza wersja wyszukanej notatki":
                            text_to_translate = st.session_state["search_note_text"]
                        else:
                            text_to_translate = st.session_state["search_note_text_corrected"]

                        selected_lang_display = st.selectbox(
                            "Wybierz język docelowy do tłumaczenia:",
                            language_display,
                            key="search_target_language_select"
                        )
                        selected_lang_code = selected_lang_display.split("(")[-1].replace(")", "").strip()
                        selected_lang_prompt = tts_languages[selected_lang_code][1]

                        if st.button("Przetłumacz na wybrany język", key="search_translate_any"):
                            openai_client = get_openai_client()
                            response = openai_client.chat.completions.create(
                                model="gpt-4o",
                                messages=[
                                    {"role": "system", "content": f"Jesteś tłumaczem. Przetłumacz poniższy tekst na {selected_lang_prompt}, zachowując sens i styl oryginału."},
                                    {"role": "user", "content": text_to_translate},
                                ],
                                max_tokens=1024,
                            )
                            st.session_state["search_translated_text_any"] = response.choices[0].message.content
                            st.session_state["search_translated_lang_code"] = selected_lang_code

                        if "search_translated_text_any" in st.session_state:
                            st.session_state["search_translated_text_any"] = st.text_area(
                                f"Tłumaczenie na {tts_languages.get(st.session_state.get('search_translated_lang_code', 'en'), ('Wybrany język',))[0]} (możesz edytować):",
                                value=st.session_state["search_translated_text_any"],
                                key="search_translated_text_any_area"
                            )

                            voice_options = ["alloy", "onyx", "echo", "fable", "nova", "shimmer"]
                            selected_voice_any = st.selectbox(
                                "Wybierz typ/rodzaj głosu do syntezy mowy:",
                                options=voice_options,
                                key="search_tts_voice_select_any"
                            )

                            if st.button("Wygeneruj audio z tłumaczenia na wybrany język", key="search_tts_any"):
                                openai_client = get_openai_client()
                                tts_response = openai_client.audio.speech.create(
                                    model="tts-1",
                                    voice=selected_voice_any,
                                    input=st.session_state["search_translated_text_any"],
                                    response_format="mp3",
                                    speed=1.0,
                                )
                                st.session_state["search_tts_any_audio"] = tts_response.content

                            if "search_tts_any_audio" in st.session_state:
                                st.audio(st.session_state["search_tts_any_audio"], format="audio/mp3")
                    else:
                        st.info("Brak wersji wyszukanej notatki do tłumaczenia. Edytuj lub popraw notatkę powyżej.")