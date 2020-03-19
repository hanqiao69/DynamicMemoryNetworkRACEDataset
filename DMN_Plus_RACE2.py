# from babi_loader import BabiDataset, pad_collate
from RACE_dataloader2 import RACEDataset, pad_collate
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data import DataLoader
import numpy as np

MAX_ANS_LEN = 15

def position_encoding(embedded_sentence):
    '''
    embedded_sentence.size() -> (#batch, #sentence, #token, #embedding)
    l.size() -> (#sentence, #embedding)
    output.size() -> (#batch, #sentence, #embedding)
    '''
    _, _, slen, elen = embedded_sentence.size()

    l = [[(1 - s / (slen - 1)) - (e / (elen - 1)) * (1 - 2 * s / (slen - 1)) for e in range(elen)] for s in range(slen)]
    l = torch.FloatTensor(l).requires_grad_()
    l = l.unsqueeze(0)  # for #batch
    l = l.unsqueeze(1)  # for #sen
    l = l.expand_as(embedded_sentence)
    weighted = embedded_sentence * l # .cuda()
    return torch.sum(weighted, dim=2).squeeze(2)  # sum with tokens


class AttentionGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wr = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(hidden_size, hidden_size)
        self.W = nn.Linear(input_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fact, C, g):
        '''
        fact.size() -> (#batch, #hidden = #embedding)
        c.size() -> (#hidden, ) -> (#batch, #hidden = #embedding)
        r.size() -> (#batch, #hidden = #embedding)
        h_tilda.size() -> (#batch, #hidden = #embedding)
        g.size() -> (#batch, )
        '''

        r = self.sigmoid(self.Wr(fact) + self.Ur(C))
        h_tilda = torch.tanh(self.W(fact) + r * self.U(C))
        g = g.unsqueeze(1).expand_as(h_tilda)
        h = g * h_tilda + (1 - g) * C
        return h


class AttentionGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.AGRUCell = AttentionGRUCell(input_size, hidden_size)

    def forward(self, facts, G):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        fact.size() -> (#batch, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        g.size() -> (#batch, )
        C.size() -> (#batch, #hidden)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        C = torch.zeros(self.hidden_size, requires_grad=True) # .cuda()
        for sid in range(sen_num):
            fact = facts[:, sid, :]
            g = G[:, sid]
            if sid == 0:
                C = C.unsqueeze(0).expand_as(fact)
            C = self.AGRUCell(fact, C, g)
        return C


class EpisodicMemory(nn.Module):
    def __init__(self, hidden_size):
        super(EpisodicMemory, self).__init__()
        self.AGRU = AttentionGRU(hidden_size, hidden_size)
        self.z1 = nn.Linear(4 * hidden_size, hidden_size)
        self.z2 = nn.Linear(hidden_size, 1)
        self.next_mem = nn.Linear(3 * hidden_size, hidden_size)

    def make_interaction(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        z.size() -> (#batch, #sentence, 4 x #embedding)
        G.size() -> (#batch, #sentence)
        '''
        batch_num, sen_num, embedding_size = facts.size()
        questions = questions.expand_as(facts)
        prevM = prevM.expand_as(facts)

        z = torch.cat([
            facts * questions,
            facts * prevM,
            torch.abs(facts - questions),
            torch.abs(facts - prevM)
        ], dim=2)

        z = z.view(-1, 4 * embedding_size)

        G = torch.tanh(self.z1(z))
        G = self.z2(G)
        G = G.view(batch_num, -1)
        G = F.softmax(G, dim=-1)

        return G

    def forward(self, facts, questions, prevM):
        '''
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #sentence = 1, #hidden)
        prevM.size() -> (#batch, #sentence = 1, #hidden = #embedding)
        G.size() -> (#batch, #sentence)
        C.size() -> (#batch, #hidden)
        concat.size() -> (#batch, 3 x #embedding)
        '''
        G = self.make_interaction(facts, questions, prevM)
        C = self.AGRU(facts, G)
        concat = torch.cat([prevM.squeeze(1), C, questions.squeeze(1)], dim=1)
        next_mem = F.relu(self.next_mem(concat))
        next_mem = next_mem.unsqueeze(1)
        return next_mem


class QuestionModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(QuestionModule, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, questions, word_embedding):
        '''
        questions.size() -> (#batch, #token)
        word_embedding() -> (#batch, #token, #embedding)
        gru() -> (1, #batch, #hidden)
        '''
        questions = word_embedding(questions)
        _, questions = self.gru(questions)
        questions = questions.transpose(0, 1)
        return questions


class InputModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(InputModule, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, contexts, word_embedding):
        '''
        contexts.size() -> (#batch, #sentence, #token)
        word_embedding() -> (#batch, #sentence x #token, #embedding)
        position_encoding() -> (#batch, #sentence, #embedding)
        facts.size() -> (#batch, #sentence, #hidden = #embedding)
        '''
        batch_num, sen_num, token_num = contexts.size()

        contexts = contexts.view(batch_num, -1)
        contexts = word_embedding(contexts)

        contexts = contexts.view(batch_num, sen_num, token_num, -1)
        contexts = position_encoding(contexts)
        contexts = self.dropout(contexts)

        h0 = torch.zeros(2, batch_num, self.hidden_size, requires_grad=True) # .cuda()
        facts, hdn = self.gru(contexts, h0)
        facts = facts[:, :, :hidden_size] + facts[:, :, hidden_size:]
        return facts


class AnswerModule(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(AnswerModule, self).__init__()
        self.z = nn.Linear(2 * hidden_size, vocab_size*MAX_ANS_LEN)
        self.dropout = nn.Dropout(0.1)

    def forward(self, M, questions):
        M = self.dropout(M)
        concat = torch.cat([M, questions], dim=2).squeeze(1)
        z = self.z(concat)
        z = z.view(questions.shape[0], MAX_ANS_LEN, -1)
        return z


class DMNPlus(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_hop=3, qa=None):
        super(DMNPlus, self).__init__()
        self.num_hop = num_hop
        self.qa = qa
        self.word_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0, sparse=True)
        init.uniform_(self.word_embedding.state_dict()['weight'], a=-(3 ** 0.5), b=3 ** 0.5)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

        self.input_module = InputModule(vocab_size, hidden_size)
        self.question_module = QuestionModule(vocab_size, hidden_size)
        self.memory = EpisodicMemory(hidden_size)
        self.answer_module = AnswerModule(vocab_size, hidden_size)

    def forward(self, contexts, questions):
        '''
        contexts.size() -> (#batch, #sentence, #token) -> (#batch, #sentence, #hidden = #embedding)
        questions.size() -> (#batch, #token) -> (#batch, 1, #hidden)
        '''
        facts = self.input_module(contexts, self.word_embedding)
        questions = self.question_module(questions, self.word_embedding)
        M = questions
        for hop in range(self.num_hop):
            M = self.memory(facts, questions, M)
        preds = self.answer_module(M, questions)
        return preds

    def interpret_indexed_tensor(self, var):
        if len(var.size()) == 3:
            # var -> n x #sen x #token
            for n, sentences in enumerate(var):
                for i, sentence in enumerate(sentences):
                    s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                    print(f'{n}th of batch, {i}th sentence, {s}')
        elif len(var.size()) == 2:
            # var -> n x #token
            for n, sentence in enumerate(var):
                s = ' '.join([self.qa.IVOCAB[elem.data[0]] for elem in sentence])
                print(f'{n}th of batch, {s}')
        elif len(var.size()) == 1:
            # var -> n (one token per batch)
            for n, token in enumerate(var):
                s = self.qa.IVOCAB[token.data[0]]
                print(f'{n}th of batch, {s}')

    def get_loss(self, contexts, questions, targets, options):
        output = self.forward(contexts, questions)
        # print('debug output: ', output.shape)
        preds = F.softmax(output, dim=-1)
        _, pred_ids = torch.max(preds, dim=2)
        acc = self.choose_option(pred_ids, targets, options)

        targets = targets.view(-1,1).squeeze(1)
        output = output.view(targets.shape[0], -1)
        # print('debug msg: ', output.shape, targets.shape)
        loss = self.criterion(output, targets.long())
        reg_loss = 0
        for param in self.parameters():
            reg_loss += 0.001 * torch.sum(param * param)
        # preds = F.softmax(output, dim=-1)
        # _, pred_ids = torch.max(preds, dim=1)

        # acc = self.choose_option(pred_ids, targets, options)
        # corrects = (pred_ids.data == targets.data)
        # acc = torch.mean(corrects.float())
        return loss + reg_loss, acc

    def choose_option(self, pred_ids, targets, options):
        # print('debug shapes: ', pred_ids.shape, targets.shape, options.shape)
        correct = 0.
        batch_size = options.shape[0]
        pred_ids = pred_ids.unsqueeze(1)
        # print('debug pred_ids: ', pred_ids.shape)
        pred_ids_expanded = pred_ids.repeat(1,4,1)
        # print('debug pred_ids_expanded: ', pred_ids_expanded.shape)
        diffs = pred_ids_expanded - options
        diffs_summed = torch.sum(diffs, dim=2)
        _, option_idxs = torch.max(diffs_summed, dim=1)
        # print('debug option_idxs: ', option_idxs)
        for i in range(batch_size):
            idx = int(option_idxs[i])
            selected = options[i,idx]
            if torch.all(targets[i].eq(selected)):
                correct += 1.
        return correct / batch_size


        # squeezed_dim = targets.shape[0]
        # options = options.view(-1, squeezed_dim)
        # losses = [self.criterion(output, option) for option in options]
        # # options = options.squeeze(0).long()
        # # print('debug shape: ', preds.shape, targets.shape, options.shape)
        # # losses = [self.criterion(preds, option) for option in options]
        # # min_loss_idx = np.argmin(losses)
        # # # print('debug target and option: ', targets, options[min_loss_idx], options)
        # # if torch.all(targets.eq(options[min_loss_idx])):
        # # # if (targets.data == options[min_loss_idx].data) == MAX_ANS_LEN:
        # #     return 1.
        # # else: return 0.


if __name__ == '__main__':
    dset_dict, vocab_dict = {}, {}
    # for task_id in range(1, 21):
    #     dset_dict[task_id] = BabiDataset(task_id)
    #     vocab_dict[task_id] = len(dset_dict[task_id].QA.VOCAB)
    for run in range(10):
        # for task_id in range(1, 21):
            dset = RACEDataset(mode='train')
            vocab_size = len(dset.QA.VOCAB)
            hidden_size = 80
            print('debug model size: ', vocab_size, hidden_size)
            model = DMNPlus(hidden_size, vocab_size, num_hop=3, qa=dset.QA)
            model # .cuda()
            early_stopping_cnt = 0
            early_stopping_flag = False
            best_acc = 0
            optim = torch.optim.Adam(model.parameters())

            for epoch in range(256):
                dset.set_mode('train')
                train_loader = DataLoader(dset, batch_size=2, shuffle=True, collate_fn=pad_collate)

                model.train()
                if not early_stopping_flag:
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(train_loader):
                        optim.zero_grad()
                        contexts, questions, answers, options = data
                        # print('debug msg data shape: ', contexts.shape, questions.shape, answers.shape)
                        batch_size = contexts.size()[0]
                        contexts = contexts.long() # .cuda()
                        questions = questions.long() # .cuda()
                        answers = answers # .cuda()

                        loss, acc = model.get_loss(contexts, questions, answers, options)
                        loss.backward()
                        total_acc += acc * batch_size
                        cnt += batch_size

                        if batch_idx % 20 == 0:
                            print(f'Epoch {epoch}] [Training] loss : {loss.item(): {10}.{8}}, '
                                  f'acc : {total_acc / cnt: {5}.{4}}, batch_idx : {batch_idx}')
                        optim.step()

                    dset.set_mode('valid')
                    valid_loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=pad_collate)

                    model.eval()
                    total_acc = 0
                    cnt = 0
                    for batch_idx, data in enumerate(valid_loader):
                        contexts, questions, answers, options = data
                        batch_size = contexts.size()[0]
                        contexts = contexts.long() # .cuda()
                        questions = questions.long() # .cuda()
                        answers = answers # .cuda()

                        _, acc = model.get_loss(contexts, questions, answers, options)
                        total_acc += acc * batch_size
                        cnt += batch_size

                    total_acc = total_acc / cnt
                    if total_acc > best_acc:
                        best_acc = total_acc
                        best_state = model.state_dict()
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1
                        if early_stopping_cnt > 20:
                            early_stopping_flag = True

                    print(f'[Run {run}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}')
                    with open('log_b128.txt', 'a') as fp:
                        fp.write(
                            f'[Run {run}, Epoch {epoch}] [Validate] Accuracy : {total_acc: {5}.{4}}' + '\n')
                    if total_acc == 1.0:
                        break
                else:
                    print(
                        f'[Run {run}] Early Stopping at Epoch {epoch}, Valid Accuracy : {best_acc: {5}.{4}}')
                    break

            dset.set_mode('test')
            test_loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=pad_collate)
            test_acc = 0
            cnt = 0

            for batch_idx, data in enumerate(test_loader):
                contexts, questions, answers, options = data
                batch_size = contexts.size()[0]
                contexts = contexts.long() # .cuda()
                questions = questions.long() # .cuda()
                answers = answers # .cuda()

                model.load_state_dict(best_state)
                _, acc = model.get_loss(contexts, questions, answers, options)
                test_acc += acc * batch_size
                cnt += batch_size
            print(f'[Run {run}, Epoch {epoch}] [Test] Accuracy : {test_acc / cnt: {5}.{4}}')
            os.makedirs('models', exist_ok=True)
            with open(f'models/epoch{epoch}_run{run}_acc{test_acc / cnt}.pth', 'wb') as fp:
                torch.save(model.state_dict(), fp)
            with open('log_b128.txt', 'a') as fp:
                fp.write(f'[Run {run}, Epoch {epoch}] [Test] Accuracy : {total_acc: {5}.{4}}' + '\n')
