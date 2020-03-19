from DMN_Plus_RACE2 import *

BATCH_SIZE = 2
CHECKPOINT_PATH = './checkpt.pth'

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
                train_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)

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
                        if batch_idx % 100 == 0:
                            torch.save(model.state_dict(), CHECKPOINT_PATH)

                    dset.set_mode('valid')
                    valid_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

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
            test_loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
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
