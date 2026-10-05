mod model;
mod ops;
mod safetensors;
mod sampler;
mod tokenizer;

use model::{Gpt2, KvCache, MAX_POS};
use safetensors::SafeTensors;
use sampler::Rng;
use std::io::Write;
use std::str::FromStr;
use std::time::Instant;
use tokenizer::Tokenizer;

const END_OF_TEXT: u32 = 50256; // GPT-2's "the document is over" token

const USAGE: &str = "\
GPT-2 (124M) text generation on the CPU.

usage: gpt2infrencemodel [OPTIONS] [PROMPT]

  PROMPT               text to continue (put it in quotes)
  -n, --tokens N       how many new tokens to make         [default: 100]
  -k, --top-k K        pick only from the K likeliest      [default: 40]
  -t, --temp T         randomness: low = safe, high = wild  [default: 0.8]
  -s, --seed S         same seed + same options = same text [default: clock]
      --greedy         always pick the likeliest token (same as -k 1)
      --bench          speed test: greedy, 50 tokens, prints tokens/sec
  -h, --help           show this help

example: cargo run --release -- \"The meaning of life is\" -n 60 -t 0.7";

// Everything the user can choose on the command line.
struct Args {
    prompt: String,
    n_tokens: usize,
    top_k: usize,
    temperature: f32,
    seed: Option<u64>, // None = pick one from the clock
    bench: bool,
}

fn main() {
    // print errors as plain text and exit with code 1
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = parse_args()?;
    if args.bench {
        args.prompt = "The quick brown fox jumps over the lazy".to_string();
        args.n_tokens = 50;
        args.top_k = 1;
    }

    let load_start = Instant::now();
    let tokenizer = Tokenizer::load("model/vocab.json", "model/merges.txt")
        .map_err(|e| format!("{e} (run ./download_model.sh first)"))?;
    // load the weights into the model; the raw file is dropped after this block
    let gpt2 = {
        let st = SafeTensors::load("model/model.safetensors")
            .map_err(|e| format!("{e} (run ./download_model.sh first)"))?;
        Gpt2::load(&st)?
    };
    eprintln!("[model loaded in {:.2}s]", load_start.elapsed().as_secs_f64());

    // an empty prompt starts from the "new document" token, like GPT-2 training did
    let mut prompt_ids = tokenizer.encode(&args.prompt)?;
    if prompt_ids.is_empty() {
        prompt_ids.push(END_OF_TEXT);
    }
    if prompt_ids.len() >= MAX_POS {
        return Err(format!("prompt is {} tokens; GPT-2 reads at most {}", prompt_ids.len(), MAX_POS - 1).into());
    }

    let seed = match args.seed {
        Some(s) => s,
        None => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos() as u64,
    };
    let mut rng = Rng::new(seed);

    print!("{}", args.prompt);
    let start = Instant::now();
    let ids = generate(&gpt2, &tokenizer, &prompt_ids, args.n_tokens, args.top_k, args.temperature, &mut rng, true)?;
    let secs = start.elapsed().as_secs_f64();
    println!();

    if args.bench {
        println!("greedy, {} tokens in {secs:.2}s = {:.1} tokens/sec", ids.len(), ids.len() as f64 / secs);
    } else {
        eprintln!(
            "[{} tokens in {secs:.2}s = {:.1} tokens/sec | k={} temp={} seed={seed}]",
            ids.len(),
            ids.len() as f64 / secs,
            args.top_k,
            args.temperature
        );
    }
    Ok(())
}

// Reads the command line into Args, starting from the defaults.
fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        prompt: "The quick brown fox jumps over the lazy".to_string(),
        n_tokens: 100,
        top_k: 40,
        temperature: 0.8,
        seed: None,
        bench: false,
    };
    let mut prompt_given = false;

    // skip(1): the first item is the program's own name
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "-n" | "--tokens" => args.n_tokens = value(&mut it, &arg)?,
            "-k" | "--top-k" => args.top_k = value(&mut it, &arg)?,
            "-t" | "--temp" => args.temperature = value(&mut it, &arg)?,
            "-s" | "--seed" => args.seed = Some(value(&mut it, &arg)?),
            "--greedy" => args.top_k = 1,
            "--bench" => args.bench = true,
            _ if arg.starts_with('-') && arg.len() > 1 => {
                return Err(format!("unknown option {arg}\n\n{USAGE}"));
            }
            _ if prompt_given => {
                return Err(format!("more than one prompt given; put the prompt in quotes\n\n{USAGE}"));
            }
            _ => {
                args.prompt = arg;
                prompt_given = true;
            }
        }
    }

    if args.top_k < 1 || args.top_k > model::VOCAB {
        return Err(format!("--top-k must be 1..={}", model::VOCAB));
    }
    if !(args.temperature > 0.0) {
        return Err("--temp must be above 0".to_string());
    }
    Ok(args)
}

// Takes the next item as the value of `flag` and turns it into a number.
// T: FromStr means "any type that can be parsed from text" (usize, f32, u64, ...).
fn value<T: FromStr>(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<T, String> {
    let text = it.next().ok_or(format!("{flag} needs a value"))?;
    text.parse().map_err(|_| format!("{flag}: {text:?} is not a valid number"))
}

// The generation loop: prompt once, then one new token per step, reusing the cache.
// Returns only the newly made ids. If `stream`, prints each token as it arrives.
fn generate(
    gpt2: &Gpt2,
    tokenizer: &Tokenizer,
    prompt: &[u32],
    max_new: usize,
    k: usize,
    temperature: f32,
    rng: &mut Rng,
    stream: bool,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut cache = KvCache::new();
    let mut new_ids = Vec::new();
    if max_new == 0 {
        return Ok(new_ids);
    }

    // step 1: the whole prompt goes in and fills the cache
    let mut logits = gpt2.forward(prompt, &mut cache);

    for _ in 0..max_new {
        let id = sampler::sample_top_k(&logits, k, temperature, rng);
        if id == END_OF_TEXT {
            break;
        }
        new_ids.push(id);
        if stream {
            print!("{}", tokenizer.decode(&[id])?);
            std::io::stdout().flush()?; // show it now, don't wait for a newline
        }
        if cache.len == MAX_POS || new_ids.len() == max_new {
            break; // GPT-2 cannot read past 1024 tokens / we have enough
        }
        // next steps: only the one new token goes in
        logits = gpt2.forward(&[id], &mut cache);
    }
    Ok(new_ids)
}
