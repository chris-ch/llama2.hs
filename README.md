## llama2.hs

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure Haskell? No? Well, now you can!

This is a fork of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), implemented in pure [Haskell](https://haskell.org).

Thanks to GitHub [codespaces](https://github.com/codespaces) you don't even need to have a Haskell compiler installed on your computer. Simply create a new Code Space pointing to this repo, and you will get a brand new remote machine with all the tooling already installed and accessible directly from your browser.

## Running the llama2

You will need to install a few training sets,
for example the mini stories from [Hugging Face](https://huggingface.co/karpathy/tinyllamas/tree/main):

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

There are also bigger models, for better stories:

```shell
wget --directory-prefix=data https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```

Once a model is available, you can then run the llama2 right away: 

```shell
cabal run -- llama2 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

This is the kind of output you will get (here using the 15M model):

```text
<s>
In that little town, there was a girl named Jane. Jane had a big, colorful folder. She loved her folder very much. She took it everywhere she went.
One day, Jane went to the park with her folder. She saw a boy named Tim. Tim was sad. Jane asked, "Why are you sad, Tim?" Tim said, "I lost my toy car." Jane wanted to help Tim. She said, "Let's look for your toy car together."
They looked and looked. Then, they found the toy car under a tree. Tim was very happy. He said, "Thank you, Jane!" Jane felt good because she helped her friend. The moral of the story is that helping others can make you feel good too.
<s>
```

## License

MIT
