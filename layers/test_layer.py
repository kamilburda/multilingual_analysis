import os
from dataclasses import field, dataclass
from typing import Optional, Any
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import random
from itertools import groupby
import pdb
import re
import sys
from tqdm import tqdm
from typing import List
import logging
logging.basicConfig(level=logging.INFO)
import torch
import csv
import cld3
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

random.seed(112)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct", device_map="auto")


def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc


def Prompting(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    hidden_states, outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':64})
    # hidden_states, outputs = model.generate(**{'input_ids':inputs.input_ids})
    hidden_embed = {}
    hidden_embed_token_level = {}
    for layer_index in hidden_states:
        hidden_embed[layer_index] = tokenizer.decode(hidden_states[layer_index][0])
        hidden_embed_token_level[layer_index] = [tokenizer.decode(tok) for tok in hidden_states[layer_index][0]]
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return hidden_embed, hidden_embed_token_level, answer


def layerwise_lang_stats(hidden_embed_token_level, candidate_langs=['en', 'zh', 'es', 'ru', 'de', 'fr']):
    lang_stats = {}
    for layer in hidden_embed_token_level:
        lang_stats[layer] = {'total_count':0}
        for token in hidden_embed_token_level[layer]:
            lang_pred = cld3.get_language(token)
            if lang_pred:
                if (lang_pred.is_reliable) and (lang_pred.language in candidate_langs):
                    lang_stats[layer]['total_count'] += 1
                    if lang_pred.language in lang_stats[layer]:
                        lang_stats[layer][lang_pred.language] += 1
                    else:
                        lang_stats[layer][lang_pred.language] = 1
    return lang_stats


def layerwise_lang_distribution(lang_stats, candidate_langs):
    lang_distribution = {}
    for layer in lang_stats:
        lang_distribution[layer] = {}
        for lang in candidate_langs:
            if lang in lang_stats[layer]:
                lang_distribution[layer][lang] = lang_stats[layer][lang]/lang_stats[layer]['total_count']
            else:
                lang_distribution[layer][lang] = 0
    return lang_distribution


def layerwise_lang_distribution_bi(lang_distribution):
    lang_distribution_bi = {}
    for layer in lang_distribution:
        lang_distribution_bi[layer] = {}
        lang_distribution_bi[layer]['en'] = lang_distribution[layer]['en']
        lang_distribution_bi[layer]['non-en'] = 1 - lang_distribution_bi[layer]['en']
    return lang_distribution_bi


def layerwise_lang_distribution_th(lang_distribution):
    lang_distribution_bi = {}
    for layer in lang_distribution:
        lang_distribution_bi[layer] = {}
        lang_distribution_bi[layer]['en'] = lang_distribution[layer]['en']
        lang_distribution_bi[layer]['zh'] = lang_distribution[layer]['zh']
        lang_distribution_bi[layer]['non-en-zh'] = 1 - lang_distribution_bi[layer]['en'] - lang_distribution[layer]['zh']
    return lang_distribution_bi


def average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs=['en', 'zh', 'es', 'ru', 'de', 'fr']):
    average_lang_distribution = {}
    for lang_distribution in lst_lang_distribution:
        for layer in lang_distribution:
            if layer in average_lang_distribution:
                for lang in lang_distribution[layer]:
                    average_lang_distribution[layer][lang] += lang_distribution[layer][lang]
            else:
                average_lang_distribution[layer] = {}
                for lang in lang_distribution[layer]:
                    average_lang_distribution[layer][lang] = lang_distribution[layer][lang]
    for layer in average_lang_distribution:
        for lang in average_lang_distribution[layer]:
            average_lang_distribution[layer][lang] /= len(lst_lang_distribution)
    for lang in candidate_langs:
        if lang == 'en':
            average_lang_distribution[0][lang] = 0
        elif lang == 'zh':
            average_lang_distribution[0][lang] = 0
        else:
            average_lang_distribution[0][lang] = 1/(len(candidate_langs)-2)
    
    return average_lang_distribution


def plot_lang_distribution(lang_distribution, candidate_langs, candidate_layers):
    lang_distribution_matrix = []
    for layer in candidate_layers:
        lang_distribution_matrix.append([lang_distribution[layer][lang] for lang in candidate_langs])
    lang_distribution_matrix = np.array(lang_distribution_matrix).T
    fig, ax = plt.subplots(figsize=(11,3))
    cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
    sns.heatmap(lang_distribution_matrix, ax=ax, xticklabels=candidate_layers, yticklabels=candidate_langs, cmap=cmap)
    plt.title('Layerwise Language Distribution')
    plt.xlabel('Layer')
    plt.ylabel('Language')
    plt.show()
    plt.savefig('lang_distribution.png')


def main(argv):

    print(model)

    # Force greedy search
    model.generation_config.do_sample = False
    model.generation_config.num_beams = 1

    zh_prompts = [
    "问题：有哪些关于自我提升的好书？答案：",
    "问题：推荐一个中国苏州的旅游攻略。答案：",
    "问题：有哪些适合学习时听的歌推荐？答案：",
    "如何学好汉语口语？",
    "推荐三部文艺电影。",
    "北京有哪些特色美食？",
    "有哪些了解中国传统文化的途径？",
    "如何找到适合自己的学习方法？",
    "香港有哪些购物的好地方？",
    "如何制作一道地道的中式菜肴？",
    ]
    en_prompts = [
    "What are some popular tourist attractions in New York City?",
    "How can I improve my English writing skills?",
    "Can you recommend three must-read books from the science fiction genre?",
    "What are some effective strategies for time management?",
    "Where can I find authentic Italian cuisine in London?",
    "What are some tips for maintaining a healthy lifestyle?",
    "Can you suggest three classic movies from the 20th century?",
    "How can I develop good public speaking skills?",
    "What are some unique cultural traditions in Japan?",
    "Can you recommend three budget-friendly destinations for solo travelers?"
    ]
    de_prompts = [
    "Frage: Was sind die besten deutschen Filme? Antwort: ",
    "Frage: Was sind die besten deutschen Bücher? Antwort: ",
    "Frage: Wie kann ich meine Deutschkenntnisse verbessern? Antwort: ",
    "Kann mir jemand ein gutes Restaurant in München empfehlen?",
    "Was sind die besten Sehenswürdigkeiten in Berlin?",
    "Wie kann ich meine Deutschkenntnisse verbessern?",
    "Was sind die besten Tipps für einen guten Schlaf?",
    "Wo kann ich in Hamburg gut shoppen?",
    "Wie kann ich meine Zeit besser nutzen?",
    "Was sind die besten deutschen Serien?",
    "Was sind die besten deutschen Lieder?"
    "Kannst du drei günstige Reiseziele in Österreich für Einzelreisende empfehlen?",
    "Was sind effektive Strategien zur Stressbewältigung?"
    ]
    fr_prompts = [
    "Question: Quels sont les meilleurs restaurants à Paris? Répondre: ",
    "Question: Quels sont les meilleurs films français? Répondre: ",
    "Question: Comment améliorer ma compréhension écrite? Répondre: "
    "Quels sont les meilleurs sites touristiques à Paris?",
    "Quels sont les meilleurs livres français?",
    "Quels sont les meilleurs séries françaises?",
    "Quels sont les meilleurs chansons françaises?",
    "Comment améliorer mon français?",
    "Comment améliorer mon accent français?",
    "Comment améliorer ma compréhension orale?",
    ]
    es_prompts = [
    "Pregunta: ¿Cuáles son los mejores restaurantes en Madrid? Respuesta: ",
    "Pregunta: ¿Cuáles son las mejores películas españolas? Respuesta: ",
    "Pregunta: ¿Cómo puedo mejorar mi comprensión oral? Respuesta: ",
    "¿Cuáles son los mejores lugares turísticos en Madrid?",
    "¿Cuáles son los mejores libros españoles?",
    "¿Cuáles son las mejores series españolas?",
    "¿Cuáles son las mejores canciones españolas?",
    "¿Cómo puedo mejorar mi español?",
    "¿Cómo puedo mejorar mi acento español?",
    "¿Cómo puedo mejorar mi comprensión escrita?"
    ]
    ru_prompts = [
    "вопрос: Какие рестораны в Москве самые лучшие? Отвечать: ",
    "вопрос: Какие фильмы русские самые лучшие? Отвечать: ",
    "вопрос: Как я могу улучшить мой русский письмо? Отвечать: "
    "Какие достопримечательности в Москве самые лучшие?",
    "Какие книги русские самые лучшие?",
    "Какие сериалы русские самые лучшие?",
    "Какие песни русские самые лучшие?",
    "Как я могу улучшить мой русский?",
    "Как я могу улучшить мой русский акцент?",
    "Как я могу улучшить мой русский слух?",
    ]

    vi_prompts = [
    "Viết một đoạn văn ngắn kể về một cuộc phiêu lưu của bạn trong một ngôi làng nông thôn ở Việt Nam.",
    "Miêu tả một ngày hè tại bãi biển Nha Trang.",
    "Viết một bài thơ ngắn về cảnh đẹp của thác Bản Giốc.",
    "Hãy viết một đoạn văn mô tả một món ăn truyền thống của Việt Nam mà bạn thích nhất.",
    "Hãy viết một bài tiểu luận về tầm quan trọng của áo dài đối với văn hóa Việt Nam.",
    "Hãy viết một câu chuyện ngắn về tình bạn đặc biệt giữa hai người bạn trong một ngôi làng nhỏ ở miền núi Việt Nam.",
    "Mô tả một lễ hội truyền thống nổi tiếng ở Việt Nam mà bạn muốn tham gia.",
    "Hãy viết một đoạn văn ngắn về một danh lam thắng cảnh nổi tiếng ở Huế.",
    "Hãy viết một bài thơ ngắn về những con thuyền trên sông Hàn ở Đà Nẵng.",
    "Mô tả một buổi sáng tại chợ Bến Thành ở Sài Gòn.",
    ]
    th_prompts = [
    "คุณชอบกินอาหารไทยประเภทใดที่สุดและเพราะอะไร?",
    "คุณเคยไปเที่ยวที่ไทยมาก่อนหรือไม่? ถ้าใช่ สถานที่ไหนที่คุณแนะนำให้คนอื่นไปเยือน?",
    "คุณคิดว่าวัฒนธรรมและประเพณีในประเทศไทยมีความสำคัญอย่างไร?",
    "คุณชื่นชอบเทศกาลไหนในประเทศไทยที่สนุกที่สุดและทำไม?",
    "ถ้าคุณมีโอกาสไปเยือนจังหวัดใดของประเทศไทย คุณจะเลือกไปที่ไหนและเพราะเหตุใด?",
    "คุณเคยลองเรียนร้องเพลงไทยหรือเต้นรำไทยมาก่อนหรือไม่? ถ้ายังไม่เคย คุณสนใจลองทำในอนาคตหรือไม่?",
    "หากคุณได้สัมผัสวิถีชีวิตของชาวไทยตั้งแต่ต้นจนปลาย อะไรบ้างที่คุณคิดว่าจะต้องปรับเปลี่ยนหรือปรับปรุง?",
    "คุณเคยเรียนรู้ภาษาไทยมาก่อนหรือไม่? ถ้าเคยคุณคิดว่าภาษาไทยมีความยากหรือง่ายอย่างไร?",
    "คุณมีเคล็ดลับในการท่องเที่ยวในประเทศไทยหรือไม่? ถ้ามีคุณสามารถบอกเราได้ไหม?",
    "คุณเคยพบกับความเปลี่ยนแปลงในประเทศไทยในช่วง 5 ปีที่ผ่านมาหรือไม่? ถ้าเคยคุณคิดว่ามีอะไรที่ทำให้คุณประทับใจ?"
    ]
    id_prompts = [
    "Ceritakan tentang seorang anak desa yang menemukan sebuah lampu ajaib saat bermain di tepian sungai.",
    "Apa pendapat Anda tentang pengaruh media sosial terhadap remaja di Indonesia saat ini?",
    "Tulislah surat resmi kepada kepala sekolah untuk meminta izin menggunakan aula untuk kegiatan ekstrakurikuler.",
    "Gambarkan suasana pasar tradisional di Indonesia pada pagi hari.",
    "Ciptakan sebuah puisi tentang keindahan alam Indonesia yang menginspirasi kamu.",
    "Bagaimana pendapat Anda mengenai pentingnya pelestarian budaya lokal di tengah globalisasi?",
    "Buatlah ulasan tentang buku terakhir yang Anda baca yang ditulis oleh penulis Indonesia.",
    "Buatlah panduan wisata singkat untuk turis asing yang ingin mengunjungi Bali.",
    "Tulislah cerita fabel yang mengajarkan tentang pentingnya kejujuran dengan tokoh utama seekor kancil dan harimau.",
    "Tuliskan refleksi pribadi Anda tentang peran pendidikan dalam mengubah masa depan generasi muda di Indonesia.",
    ]
    ms_prompts = [
    "Tulis sebuah cerita pendek tentang seorang nelayan yang menemukan pesan dalam botol saat melaut.",
    "Apakah pandangan anda mengenai impak teknologi terhadap pendidikan di Malaysia?",
    "Huraikan suasana di Jalan Alor, Kuala Lumpur pada waktu malam.",
    "Karang surat rasmi kepada pihak berkuasa tempatan untuk melaporkan masalah sampah yang tidak dikutip di kawasan anda.",
    "Nyatakan pendapat anda tentang kepentingan memelihara warisan budaya Malaysia di era globalisasi.",
    "Tulis puisi tentang keharmonian masyarakat pelbagai kaum di Malaysia.",
    "Buat ulasan tentang sebuah novel yang anda baca baru-baru ini oleh penulis Malaysia.",
    "Buat panduan ringkas untuk pelancong asing yang ingin mengunjungi Taman Negara Pahang.",
    "Tulislah sebuah dongeng yang mengandungi pengajaran tentang kebaikan dan kesabaran dengan watak utama sang kancil.",
    "Tulis esai reflektif tentang peranan bahasa Melayu dalam memperkukuh identiti nasional Malaysia."
    ]

    prompts = zh_prompts + vi_prompts + th_prompts + id_prompts + ms_prompts

    # candidate_langs = ['en', 'zh', 'es', 'ru', 'de', 'fr']
    candidate_langs = ['en', 'zh', 'vi', 'th', 'id', 'ms']

    lst_lang_distribution = []
    for prompt in tqdm(prompts):
        hidden_embed, hidden_embed_token_level, answer = Prompting(model, prompt)
        lang_stats = layerwise_lang_stats(hidden_embed_token_level, candidate_langs)

        # if only draw english and non-english
        # lang_distribution = layerwise_lang_distribution_bi(lang_distribution)

        lang_distribution = layerwise_lang_distribution(lang_stats, candidate_langs)
        lst_lang_distribution.append(lang_distribution)
    
    # if only draw english and non-english
    # average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs=['en', 'non-en'])
    # plot_lang_distribution(average_lang_distribution, candidate_langs=['en', 'non-en'], candidate_layers=candidate_layers)
        
    # if draw all languages independently
    average_lang_distribution = average_layerwise_lang_distribution(lst_lang_distribution, candidate_langs)
    plot_lang_distribution(average_lang_distribution, candidate_langs, candidate_layers=candidate_layers)
    # pdb.set_trace()


if __name__ == "__main__":
    # sys.setprofile(tracefunc)
    main(sys.argv[1:])
