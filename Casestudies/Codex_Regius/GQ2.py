import unicodedata
import re
import pandas as pd
from collections import Counter, defaultdict
import difflib
import os
import itertools
import tkinter as tk
from pandastable import Table
from pprint import pprint

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_un = os.path.join(script_dir, "GQ2_un.txt")
file_path_no = os.path.join(script_dir, "GQ2_no.txt")

with open(file_path_un, "r", encoding="utf-8") as f:
    unnormalized_text = f.read()

with open(file_path_no, "r", encoding="utf-8") as f:
    normalized_text = f.read()

# unnormalized_text = "hafva hafua fur hava haft ver Nv uer hafva er at segia hafa fra þui hafa er Sigurdr biozst til bardaga j mot Hundings sonum. hann hafdi mikit lid ok uel uopnnat. Reginn hafde miog radagerd firir lidinu. hann hafdi suerd þat er Ridill het er hann hafde Smidat. Sigurdr bad Regin lea ser suerdit. hann gerde sua ok bad hann drepa Fafni þa er hann kæmi aftr or þessi ferd. Sigurdr het honum þui. Sidan sigldu ver sudr med landi. þa fengu ver giorningauedr stor ok kendu þat margir Hundings sonum. Sidan sigldu uer  nokkuru landhallara. þa sam uer mann æinn a biargsnỏs nokkurre er gek fram af  siofarhomrum. hann uar j heklu grænni ok blam brokum ok knefta sko a fotum upphafua ok spiot j hende. þessi madr liodar a oss ok quad. Huerir rida her ræfils hestum hafri unnar hafi glymianda. eru segl ydr siofui stokkin munu at uopnadir uind of standaz. Reginn quad i moti."
# normalized_text = "hafa hafa fur hafa haft vér  Nú vér  hafva er at segja hafa frá því hafa er Sigurðr bjósk til bardaga í mót Hundings sonum. Hann hafði mikit lið ok vel vápnat. Reginn hafði mjǫk ráðagørð fyrir liðinu. Hann hafði sverð þat er Riðill hét, er hann hafði smíðat. Sigurðr bað Reginn ljá sér sverðit. Hann gerði svá ok bað hann drepa Fáfni þá er hann kæmi aptr ór þessi ferð. Sigurðr hét honum því. Síðan sigldu vér suðr með landi. Þá fengu vér gjǫrningaveðr stór ok kenndu þat margir Hundings sonum. Síðan sigldu vér nokkuru landhallara. Þá sáum vér mann einn á bjargsnǫs nokkurri, er gekk fram af sjóvarhǫmrum. Hann var í heklu grǿnni ok blám brókum ok knéftum skóm á fótum upphafa ok spjót í hendi. Þessi maðr ljóðar á oss ok kvað: Hverir ríða hér ræfils hestum, hafri unnar hafi glymjanda Eru segl yðr sjóvi stokkin Munu at vápnaðir vind of standask? Reginn kvað í móti:"



# Step 1: Split into words
def prep(text):
    # text = unicodedata.normalize('NFKD', text).lower() #removes diacritics
    text = unicodedata.normalize('NFC', text).lower()
    text = text.replace('ꝼ', 'f')
    text = text.replace('\'', 'ç').replace('’', 'ç')
    # text = ''.join([c for c in text if not unicodedata.combining(c)])
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.replace('  ', ' ')
    text = text.strip()  # Clean leading/trailing spaces
    text = text.replace('ç', '’')
    return text

# print(repr(unnormalized_text))  # This will show special characters like spaces and punctuation
# print(repr(prep(unnormalized_text)))

unnormalized_words = prep(unnormalized_text).split()
normalized_words = prep(normalized_text).split()

# print(unnormalized_words, "\n", normalized_words, "\n")


# Step 2: Align sequences
matcher = difflib.SequenceMatcher(None, unnormalized_words, normalized_words)
# print(matcher)

# find the number of spaces, to find if there are some split words:
num_spaces_un = prep(unnormalized_text).count(' ')
num_spaces_no = prep(normalized_text).count(' ')

# print(f"There are {num_spaces_un} spaces in UN and {num_spaces_no} in NO.")

zipped_words = list(zip(normalized_words, unnormalized_words))
# print(zipped_words)

# Catch merged words, e.g. 'ekvil' for 'ek' and 'vil'.
def find_shorter_normalized(zipped, max_norm_len=2):
    results  = []
    for i, (norm, unnorm) in enumerate(zipped):
        if len(norm) < len(unnorm) and len(norm) <= max_norm_len:
            results.append((i, norm, unnorm))
    return results

# print(find_shorter_normalized(zipped_words))

# Catch words merged with prefix, e.g. 'ifra' for 'í' and 'frá'.
def find_merged_prefix(zipped):
    results  = []
    for idx, (norm, unnorm) in enumerate(zipped):
        if norm in {"í", "á"} and len(norm) < len(unnorm):
            results.append((idx, norm, unnorm))
    return results

# print(find_merged_prefix(zipped_words))

# In order to catch any e.g. split compound-words
def filter_split_compounds(zipped_words):
    filtered = [
        (normalized, unnormalized)
        for normalized, unnormalized in zipped_words
        if len(unnormalized) < len(normalized) and '’' not in unnormalized
    ]
    for norm, unnorm in filtered:
        print(f"Normalized: {norm}, Unnormalized: {unnorm}")
    return filtered

# possible_compounds = filter_split_compounds(zipped_words)
# print(possible_compounds)

# Step 3: Build the mapping. 
def generate_mapping(zipped_words):
    # Map: normword -> list of unnormalized forms
    temp_map = defaultdict(list)

    for normword, unword in zipped_words:
        temp_map[normword].append(unword)

    # Build final mapping with counts
    final_mapping = {}
    for normword, unwords in temp_map.items():
        count = len(unwords)
        freqs = Counter(unwords).items()
        final_mapping[(normword, count)] = list(freqs)

    return final_mapping

mapping = generate_mapping(zipped_words)

# Step 4: Done! 
# print(mapping)

# Filtering mappings where the value contains 'v', 'u', or 'f'. 
filtered = [(a, b) for (a, b) in mapping.items() if a and any(c in a[0] for c in 'vuúf')]

# Print the filtered results
# print(filtered)

filtered_v = [(a, b) for (a, b) in filtered if 'v' in a[0]]
filtered_u = [(a, b) for (a, b) in filtered if any(c in a[0] for c in 'uú')]
filtered_f = [(a, b) for (a, b) in filtered if 'f' in a[0][1:]]   # at least one 'f' in positions 1…

# print("Words with <v>:")
# for (norm_word, count), un_words in filtered_v:
#     un_str = ', '.join(f"'{uw}' ({uc})" for uw, uc in un_words)
#     print(f"'{norm_word}' ({count}): {un_str}")

# print("\nWords with <u>:")
# for (norm_word, count), un_words in filtered_u:
#     un_str = ', '.join(f"'{uw}' ({uc})" for uw, uc in un_words)
#     print(f"'{norm_word}' ({count}): {un_str}")

# print("\nWords with <f>:")
# for (norm_word, count), un_words in filtered_f:
#     un_str = ', '.join(f"'{uw}' ({uc})" for uw, uc in un_words)
#     print(f"'{norm_word}' ({count}): {un_str}")


def overview_v_representations(pairs):
    counter = Counter()
    words   = defaultdict(list)
    clusters = {'fu', 'fv'} 

    for (norm, _), un_list in pairs:
        for un_str, count in un_list:
            i = norm_i = 0
            while i < len(un_str) and norm_i < len(norm):
                found_cluster = False

                # --- cluster case: look for 'fu' under a 'v' ---
                for cluster in clusters:
                    if un_str[i:i+len(cluster)] == cluster and norm[norm_i] == 'v':
                        # make sure the normalized word doesn’t really continue with that second char
                        next_norm = norm[norm_i+1:norm_i+2]
                        if next_norm != cluster[1]:
                            counter[cluster] += count
                            words[cluster].append((norm, un_str, count))
                            i += len(cluster)
                            norm_i += 1
                            found_cluster = True
                            break

                if not found_cluster:
                    # --- single-char case: plain 'v' or 'u' under 'v' ---
                    if norm[norm_i] == 'v':
                        rep = un_str[i]
                        counter[rep] += count
                        words[rep].append((norm, un_str, count))

                    i += 1
                    norm_i += 1

    return dict(counter), dict(words)

v_representations = overview_v_representations(filtered_v)
# print(v_representations)

count_v, words_v = v_representations

rows = []
for rep in count_v:
    first = True
    for normword, unword, count in words_v[rep]:
        rows.append({
            "Representation": rep if first else "",
            "Word Amount": count_v[rep] if first else "",
            "Normalized": normword,
            "Unnormalized": unword,
            "Count": count
        })
        first = False

v_df = pd.DataFrame(rows)
# print("\nUnnormalized representation of <v>:", "\n", v_df)


def overview_u_representations(pairs):
    counter = Counter()
    words = defaultdict(list)

    for (norm, _), un_list in pairs:
        for un_str, count in un_list:
            for n_c, u_c in zip(norm, un_str):
                if n_c in ('u', 'ú'):
                    counter[u_c] += count
                    words[u_c].append((norm, un_str, count))

    return dict(counter), dict(words)

u_representations = overview_u_representations(filtered_u)

count_u, words_u = u_representations

rows = []
for rep in count_u:
    first = True
    for normword, unword, count in words_u[rep]:
        rows.append({
            "Representation": rep if first else "",
            "Word Amount": count_u[rep] if first else "",
            "Normalized": normword,
            "Unnormalized": unword,
            "Count": count
        })
        first = False

u_df = pd.DataFrame(rows)

# print("\nUnnormalized representation of <u>:", "\n", u_df)


def overview_f_representations(pairs):
    counter = Counter()
    words   = defaultdict(list)
    clusters = {'fv', 'fu'}

    for (norm, _), un_list in pairs:
        for un_str, count in un_list:
            i = norm_i = 0
            while i < len(un_str) and norm_i < len(norm):
                found_cluster = False

                # --- cluster case, e.g. 'fv' or 'fu' under a 'f' ---
                for cluster in clusters:
                    if (
                        i > 0 and
                        un_str[i:i+len(cluster)] == cluster and 
                        norm[norm_i] == 'f'
                    ):
                        next_norm = norm[norm_i+1:norm_i+2]
                        if next_norm != cluster[1]:
                            counter[cluster] += count
                            words[cluster].append((norm, un_str, count))
                            i += len(cluster)
                            norm_i += 1
                            found_cluster = True
                            break

                if not found_cluster:
                    # --- single‐char case under 'f' (catch f, v, u, etc.) ---
                    if i > 0 and norm[norm_i] == 'f':
                        rep = un_str[i]
                        counter[rep] += count
                        words[rep].append((norm, un_str, count))

                    i += 1
                    norm_i += 1

    return dict(counter), dict(words)

f_representations = overview_f_representations(filtered_f)
# print(f_representations)

count_f, words_f = f_representations

rows = []
for rep in count_f:
    first = True
    for normword, unword, count in words_f[rep]:
        rows.append({
            "Representation": rep if first else "",
            "Word Amount": count_f[rep] if first else "",
            "Normalized": normword,
            "Unnormalized": unword,
            "Count": count
        })
        first = False

f_df = pd.DataFrame(rows)

# print("\nUnnormalized representation of <f>:", "\n", f_df)


# Wholistic overview of representations:
allowed = {'v', 'u', 'ú', 'f', 'fv', 'fu', 'o'}
all_reps = (set(count_v) | set(count_u) | set(count_f)) & allowed

rows = []
for rep in all_reps:
    rows.append({
        "Representation": rep,
        "<v>": count_v.get(rep, 0),
        "<u>": count_u.get(rep, 0),
        "<f>": count_f.get(rep, 0),
    })

tot_df = pd.DataFrame(rows)

# 1) per‐row totals
tot_df['Total'] = tot_df[['<v>','<u>','<f>']].sum(axis=1)

# 2) append one “grand total” row
tot_df = pd.concat([
    tot_df,
    pd.DataFrame([{
        'Representation':'Total',
        **tot_df[['<v>','<u>','<f>','Total']].sum().to_dict()
    }])
], ignore_index=True)

common_reps = (set(words_v) & set(words_u) & set(words_f)) & allowed

rows = []
for rep in (set(count_v) | set(count_u) | set(count_f)) & allowed:
    # pull word‐lists, defaulting to empty list if none
    v_list = words_v.get(rep, [])
    u_list = words_u.get(rep, [])
    f_list = words_f.get(rep, [])
    first = True

    for v_item, u_item, f_item in itertools.zip_longest(
            v_list, u_list, f_list, fillvalue=("", "", 0)
    ):
        norm_v, un_v, cnt_v = v_item
        norm_u, un_u, cnt_u = u_item
        norm_f, un_f, cnt_f = f_item

        rows.append({
            "Representation": rep if first else "",
            "<v>": count_v.get(rep, ""),
            "<u>": count_u.get(rep, ""),
            "<f>": count_f.get(rep, ""),
            "v_norm": norm_v,
            "v_un":   un_v,
            "v_cnt":  cnt_v,
            "u_norm": norm_u,
            "u_un":   un_u,
            "u_cnt":  cnt_u,
            "f_norm": norm_f,
            "f_un":   un_f,
            "f_cnt":  cnt_f,
        })
        first = False

det_df = pd.DataFrame(rows)

# print("\nOverview of Unnormalized Representation of <v u f>:", "\n", tot_df)
# print("\nDetailed breakdown of <v u f> by word:\n", det_df)

def show_df_tk(df):
    root = tk.Tk()
    root.title("DataFrame Viewer")
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)
    pt = Table(frame, dataframe=df, showstatusbar=True)
    pt.show()
    root.mainloop()

# …after tot_df is ready:
show_df_tk(v_df)
show_df_tk(u_df)
show_df_tk(f_df)
show_df_tk(tot_df)
show_df_tk(det_df)

# print(zipped_words)
# pprint(zipped_words, width=1000)
# print(find_merged_prefix(zipped_words))
# print(find_shorter_normalized(zipped_words))
# possible_compounds = filter_split_compounds(zipped_words)