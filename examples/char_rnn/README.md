This example implements a character-level language model heavily inspired 
by [char-rnn](https://github.com/karpathy/char-rnn).
- For training, the model takes as input a text file and learns to predict the next character following
a given sequence.
- For generation, the model takes as input a seed character sequence, returns the next character distribution
from which a character is randomly sampled and this is iterated to generate a text.

By default, this uses a stack of two LSTMs. At the end of each training epoch, a checkpoint file containing
the weights is saved. This checkpoint file can then be used by the generation step.

Any text file can be used as an input, as long as it's large enough for training.
A typical example would be the
[tiny Shakespeare dataset](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

Compiling the example is done via the following command line:
```bash
dune build examples/char_rnn/char_rnn.exe
```

Training can be performed as follows:
```bash
./char_rnn.native train --train-file input.txt --checkpoint out.ckpt
```

In order to generate a sequence, the following command can be used:
```bash
./char_rnn.native sample --train-file input.txt --checkpoint out.ckpt --seed KING
```

Here is an example of generated data. Training was done on the Shakespeare dataset using only 90 epochs
(at which point the network was still learning).
> KINGE:
> I was you shall be doof, and the sun to me that the heart of them to please and mother, for your prison, and the trimimous to me come to be that I give me, and the sarrant.
> 
> LUCIO:
> A satisfied him, sir, the good and the what, and the wars of the world that have been as a sacred that why, I must thou art, is the house, and in the star and death
> And not in the man that is of the souls and make thee with me of her to the friends:
> If thou hast thou mark and all this man which we are her than the king of this drink on the hand flowers; thou art thou be against thou shalt have been to me and to the courted of death, good some show a merriard and forching well you had it for my lord, thou will you good to me.
> 
> MIRANDA:
> What shall we come, sir, that you did to I shall I swear a body have them by the daughter.
> 
> CLAUDIO:
> I may not she shall not see thee hear it was her how thy such the sendent me the noble and a sweet the all this make you and to sheme to do you bear your cause in this harm so for him, that bein


