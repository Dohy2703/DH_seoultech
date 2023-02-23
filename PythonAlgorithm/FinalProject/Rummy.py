# Dongseok Seo, Dohyeon Kim

import random
import operator
import numpy as np

class Card:
    def __init__(self, kind, number):
        self.kind = kind
        self.number = number
    def __str__(self):
        return '{0}:{1}'.format(self.kind, self.number)

class Player:
    def __init__(self):
        self.cards = []

    def printCards(self):
        for card in self.cards:
            print(card.kind + " " + str(card.number))

class Rummy:
    def __init__(self, playerCnt, distCardCnt):
        self.distCardCnt = distCardCnt
        self.playerCnt = playerCnt
        self.cards = []
        self.players = []
        self.generateCards()
        self.shuffleCards()
        self.createPlayers()
        self.remainder = []
        self.temp_cards = []
        # self.field = np.zeros((4, 13), dtype = int)
        self.field = np.full((4, 13), -1, dtype = int)
        self.field2 = np.zeros((4, 13), dtype = int)
        # self.field2 = [[0,0,0,0,5,0,0,0,0,0,0,0,0],[0,0,3,0,5,0,0,0,0,0,0,0,0],[0,0,3,0,5,0,0,0,0,0,0,0,0],[0,0,3,0,0,0,0,0,0,0,0,0,0]]


    def generateCards(self):
        self.cards = []
        kinds = ['spade', 'heart', 'diamond', 'clover']
        for i in range(4):
            for j in range(13):
                card = Card(kinds[i], j + 1)
                self.cards.append(card)
        return

    def shuffleCards(self):
        random.shuffle(self.cards)

    def createPlayers(self):
        for j in range(self.playerCnt):
            player = Player()
            self.players.append(player)

    def printCards(self):
        for card in self.cards:
            print(card.kind + "" + str(card.number))

    def playCards(self):
        for i in range(self.distCardCnt):
            for j in range(self.playerCnt):
                card = self.cards.pop()
                self.players[j].cards.append(card)

        for k in range(self.playerCnt):
            self.players[k].cards = sorted(self.players[k].cards, key=operator.attrgetter('kind', 'number'))

    def printPlayerCards(self):
        player_num = 1
        for player in self.players:
            print("player", player_num, ":\n")
            player.printCards()
            player_num += 1

    def putCards(self):
        self.reamainder = []
        player_cnt = 0

        for i in range(len(self.players)):
            if len(self.players[i].cards) == 0:
                return False
        if len(self.cards) == 0:
            return False

        for player in self.players:
            player_cnt += 1
            if player_cnt == 5:
                player_cnt = 1
            print("player:", player_cnt)
            #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ1. put the sequence of numbersㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            getter = 0
            self.temp_cards = []
            for card in player.cards:
                self.temp_cards.append((card.kind, card.number))
            # self.temp_cards = [('clover',3), ('spade',5)]
            # self.temp_cards = [('spade', 5)]
            prev_card_type = None
            prev_card_number = None
            cnt = 0
            cnt2 = 0
            end_idx = 0
            end_idx2 = 0
            for i, (curr_card_t, curr_card_n) in enumerate(self.temp_cards):
                if prev_card_type == curr_card_t and prev_card_number + 1 == curr_card_n:
                    cnt += 1
                    end_idx = i
                    if cnt >= 2:
                        getter = 1
                        break
                if prev_card_type == curr_card_t and prev_card_number + 1 != curr_card_n:
                    cnt = 0
                if prev_card_type != curr_card_t:
                    cnt = 0
                if prev_card_type != curr_card_t:
                    cnt = 0
                prev_card_type = curr_card_t
                prev_card_number = curr_card_n

            start_idx = end_idx - cnt
            # print("sta, end : ", start_idx, end_idx)
            if end_idx - start_idx >= 2:
                self.remainder = self.temp_cards[start_idx:end_idx + 1]
                del self.temp_cards[start_idx:end_idx + 1]
                for i in range(len(self.remainder)):
                    item_card = self.remainder.pop()
                    if item_card[0] == 'clover':
                        self.field[0][item_card[1]-1] = item_card[1]
                    if item_card[0] == 'diamond':
                        self.field[1][item_card[1]-1] = item_card[1]
                    if item_card[0] == 'heart':
                        self.field[2][item_card[1]-1] = item_card[1]
                    if item_card[0] == 'spade':
                        self.field[3][item_card[1]-1] = item_card[1]
                print('=' * 50)
            #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ2. put the same numberㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            self.temp_cards = sorted(self.temp_cards, key=lambda temp_card: temp_card[1])
            prev_card_number = None
            for i, (curr_card_t, curr_card_n) in enumerate(self.temp_cards):
                if prev_card_number == curr_card_n:
                    cnt2 += 1
                    if cnt2>=2:
                        end_idx2 = i
                elif (cnt2 >= 2 and prev_card_number != curr_card_n):
                    end_idx2 = i-1
                    getter = 1
                    break
                else :
                    cnt2 = 0

                prev_card_number = curr_card_n

            start_idx2 = end_idx2 - cnt2
            if end_idx2 - start_idx2 >= 2:
                self.remainder = self.temp_cards[start_idx2:end_idx2 + 1]
                del self.temp_cards[start_idx2:end_idx2 + 1]
                for i in range(len(self.remainder)):
                    item_card = self.remainder.pop()
                    if item_card[0] == 'clover':
                        self.field2[0][item_card[1] - 1] = item_card[1]
                    if item_card[0] == 'diamond':
                        self.field2[1][item_card[1] - 1] = item_card[1]
                    if item_card[0] == 'heart':
                        self.field2[2][item_card[1] - 1] = item_card[1]
                    if item_card[0] == 'spade':
                        self.field2[3][item_card[1] - 1] = item_card[1]
                print('=' * 50)

            # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ3. attach to sequence of numbersㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            if(getter == 0):
                for card in self.temp_cards:
                    for i in range(0,2):
                        if card[0] == 'clover':
                            for j in range(0,12):
                                if self.field[0][j] == card[1] - 1:
                                    self.field[0][j+1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue
                                elif self.field[0][j] == card[1] + 1 :
                                    self.field[0][j-1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue
                        if card[0] == 'diamond':
                            for j in range(0,12):
                                if self.field[1][j] == card[1] - 1:
                                    self.field[1][j+1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue
                                elif self.field[1][j] == card[1] + 1:
                                    self.field[1][j-1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue


                        if card[0] == 'heart':
                            for j in range(0,12):
                                if self.field[2][j] == card[1] - 1:
                                    self.field[2][j+1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue
                                elif self.field[2][j] == card[1] + 1:
                                    self.field[2][j-1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue

                        if card[0] == 'spade':
                            for j in range(0,12):
                                if self.field[3][j] == card[1] - 1:
                                    self.field[3][j+1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue
                                elif self.field[3][j] == card[1] + 1:
                                    self.field[3][j-1] = card[1]
                                    try:
                                        idx = self.temp_cards.index(card)
                                        self.temp_cards.pop(idx)
                                        getter += 1
                                    except:
                                        continue

            # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ4. attach to same numberㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#

            # print(self.temp_cards[0][1])
            idx_list = []
            if (getter == 0):  
                for card in range(len(self.temp_cards)):
                    for i in range(13):
                        same_col_cnt = 0
                        same_num_cnt = 0
                        for j in range(4):
                            if self.field2[j][i] != 0:
                                same_num_cnt += 1
                                same_col_cnt += j
                                if same_num_cnt == 3 and i+1==self.temp_cards[card][1]:
                                    getter += 1
                                    # print('1')
                                    self.field2[6-same_col_cnt][i]=self.temp_cards[card][1]
                                    idx_list.append(card)
                                    print(idx_list)
            idx_list.sort(reverse=True)
            for item in idx_list:
                self.temp_cards.pop(item)

            #ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
            if (getter == 0 and len(self.cards) != 0):
                a = self.cards.pop()
                player.cards.append(a)
                self.temp_cards.append((a.kind,a.number))
                if a.kind == 'clover':
                    a.kind = '♣'
                if a.kind == 'diamond':
                    a.kind = '♦'
                if a.kind == 'heart':
                    a.kind = '♥'
                if a.kind == 'spade':
                    a.kind = '♠'
                print("player "+ str(player_cnt) + " got " + str( (a.kind, a.number) ) )
            player.cards = sorted(player.cards, key=operator.attrgetter('kind', 'number'))


            if getter != 0:
                for i in range(len(player.cards)):
                    if i in range(len(self.temp_cards)):
                        player.cards[i] = Card(self.temp_cards[i][0], self.temp_cards[i][1])
                    else :
                        player.cards.pop()

            # self.temp_cards.sort(key=lambda x: x[0])
            self.temp_cards.sort()

            for idx, card in enumerate(self.temp_cards):
                if card[0] == 'clover':
                    self.temp_cards[idx] = ('♣', card[1])
                if card[0] == 'diamond':
                    self.temp_cards[idx] = ('♦', card[1])
                if card[0] == 'heart':
                    self.temp_cards[idx] = ('♥', card[1])
                if card[0] == 'spade':
                    self.temp_cards[idx] = ('♠', card[1])


            print(f'remain : {self.field}, \n{self.field2} \ntemp_cards: {self.temp_cards}')
            print("deck left : " + str(len(self.cards)))
            print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

            if len(self.cards) == 0 or len(self.temp_cards)==0:
                self.endGame(player_cnt)
                print("end game")
                return False

    def endGame(self, player):
        player_len = []
        for i in range(len(self.players)):
            player_len.append(len(self.players[i].cards))
            print("player " + str(i + 1) + " left cards : " + str(player_len[i]))

        winner = []

        for i in range(len(self.players)):
            if len(self.players[i].cards) == min(player_len):
                winner.append(self.players[i])

        a = min(player_len)

        if (len(self.cards) != 0):
            print(f'<player {player} win>')
        elif len(winner)== 1:
            print("<player"+str(player_len.index(a)+1)+"win>")
        else :
            print("draw : ")
            for j, v in enumerate(player_len):
                if v == a:
                    print('player '+str(j+1))
        return False

playerCnt = 4
distCardCnt = 7
gameCount = 100
rummy = Rummy(playerCnt, distCardCnt)
# rummy.printCards()
rummy.playCards()
for i in range(gameCount):
    if rummy.putCards() != 0:
        rummy.putCards()
    if rummy.putCards() == 0:
        break
