## llama2.hs

<p align="center">
  <img src="assets/llama_cute.jpg" width="300" height="300" alt="Cute Llama">
</p>

Have you ever wanted to inference a baby [Llama 2](https://ai.meta.com/llama/) model in pure Haskell? No? Well, now you can!

This is a fork of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c), implemented in pure [Haskell](https://haskell.org).

Thanks to GitHub [codespaces](https://github.com/codespaces) you don't even need to have a Haskell compiler installed on your computer. Simply create a new Code Space pointing to this repo, and you will get a brand new remote machine with all the tooling already installed and accessible directly from your browser. All you need is a coffee machine nearby, because downloading all the Haskell libraries takes several minutes the first time you launch the executable.

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

Once a model is downloaded, you can then run the llama2 right away: 

```shell
cabal run -- llama2 --model-file data/stories15M.bin --temperature 0.8 --steps 256 "In that little town"
```

This is the kind of output you will get (here using the 110M model):

```text
<s>
In that little town, there was a humble house. In the house lived a kind man named Tom. Tom had a big potato farm. He loved to grow potatoes and share them with his friends.
One day, a little girl named Lily came to Tom's house. She was hungry and asked, "Can I have a potato, please?" Tom smiled and said, "Of course, Lily! I have many potatoes to offer you."
Tom gave Lily a big potato from his farm. Lily was very happy and said, "Thank you, Tom!" She went back to her home and ate the potato. It was the best potato she had ever tasted.
The next day, Lily came back to Tom's house with a big smile. She had a big bag of coins. "Tom, I want to give you this coins to say thank you for the potato," she said. Tom was very happy and thanked Lily for the coins.
From that day on, Lily and Tom became good friends. They would often talk and share potatoes from the humble little house. And they all lived happily ever after.
<s>
```

## License

MIT
