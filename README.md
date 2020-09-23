# RL_FrozenLake

# FrozenLake AI

![image](https://user-images.githubusercontent.com/58151691/94073147-a5e5a580-fdff-11ea-9046-f7231b3684ae.png)

Bu çalışmada OpenAI - Gym enviroment'ı olan "FrozenLake-v0" ile Reinforcemnt Q - Learning alıştırması yapılmıştır.

Amaç : Başlangıç noktasından başlayarak Hedef Noktaya ulaşmak.

# Enviroment'daki harflerin anlamları nelerdir?
- S : Başlangıç noktası
- F : Donmuş noktalar (Güvenli, hareket edilebilir.)
- H : (Hole : ) Çukur kısımlar (Güvenli değil, Yanıyorsunuz.)
- G : Hedef, bitiş noktası

# Ödüllendirme Sistemi (Reward) :

- Hedef noktaya(bitişe) ulaşılırsa +1, diğer durumlarda 0 yani ödül yok.
- Bölüm (episode) hedefe (bitiş noktasına) ulaşınca veya deliğe düşünce biter.

# Actionlar : 

- Sağ
- Sol
- Yukarı
- Aşağı

Actionlar stokastik şekilde tanımlanmış bunu deterministik şekle çevirmek için;

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

kullanılmıştır.

## Kullanılan Q-Learning Algoritması Formülü:

![image](https://user-images.githubusercontent.com/58151691/94074487-083fa580-fe02-11ea-93cc-fca5fdac405a.png)
