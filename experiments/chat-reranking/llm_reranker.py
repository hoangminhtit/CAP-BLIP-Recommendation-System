import pickle
import argparse
import datetime
import json
import pandas as pd
import time
import re
import os

MODEL_DICT = {"gpt-3.5-turbo": "gpt-3.5-turbo",
              "gpt-3.5-turbo-0613": "gpt-3.5-turbo-0613",
              "gpt-3.5-turbo-instruct": "gpt-3.5-turbo-instruct",
              "gpt-4": "gpt-4",
              "gpt-4-1106-preview": "gpt41106",
              "Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",
              "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b-chat-hf",
              "Llama-2-13b-chat-hf": "Llama-2-13b-chat-hf",
              "Meta-Llama-3-8B-Instruct": "Llama-3-8b-instruct",
              "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8b-instruct"
              }

DELIMITERS = {"gpt-3.5-turbo-instruct":  ["<", ">"],
              "gpt-3.5-turbo-0613":  ["<", ">"],
              "Llama-2-7b-chat-hf": ["{", "}"],
              "meta-llama/Llama-2-7b-chat-hf": ["{", "}"],
              "Llama-2-13b-chat-hf": ["{", "}"],
              "Meta-Llama-3-8B-Instruct": ["{", "}"],
              "meta-llama/Meta-Llama-3-8B-Instruct": ["{", "}"],
              "gpt-4-1106-preview": ["<", ">"]
              }

DEFAULT_DELIMITERS = ["<", ">"]

HF_MODEL_REPOS = {
    # "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    # "Llama-2-13b-chat-hf": "meta-llama/Llama-2-13b-chat-hf",
    "Meta-Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
}

MODELS = {"gpt-3.5-turbo": "optimized for chat. It should be the latest model available but I am not sure about OpenAI updates policies",
          "gpt-3.5-turbo-0613": "Snapshot of gpt-3.5-turbo from June 13th 2023 with function calling data. Unlike gpt-3.5-turbo, this model will not receive updates, and will be deprecated 3 months after a new version is released.",
          "gpt-3.5-turbo-instruct": "Similar capabilities as text-davinci-003 but compatible with legacy Completions endpoint and not Chat Completions."
          }


class PromptLLM:
    def __init__(self,
                 llm_name: str,
                 prompts: list,
                 itemname_to_id: dict
                 ):
        self.llm_name = llm_name
        self.prompts = prompts
        self.itemname_to_id = itemname_to_id
        self.output = None

    @staticmethod
    def parse_response(output: str,
                       itemname_to_id: dict
                       ) -> list:

        lines = output.splitlines()
        reranked_recs = []
        for line in lines:
            try:
                if len(line.split("-> ")) > 1:
                    item_name = line.split("-> ", 1)[1]
                    item_name = PromptLLM.normalize_item_name(item_name)
                    if item_name in itemname_to_id:
                        reranked_recs.append(itemname_to_id[item_name])
                    continue

                match = re.match(r"^\s*\d+\s*[\.\-\)]\s*(.+?)\s*$", line)
                if match:
                    item_name = PromptLLM.normalize_item_name(match.group(1))
                    if item_name in itemname_to_id:
                        reranked_recs.append(itemname_to_id[item_name])
            except Exception as e:
                continue
        return list(dict.fromkeys(reranked_recs))

    @staticmethod
    def normalize_item_name(item_name: str) -> str:
        item_name = item_name.strip()
        item_name = item_name.strip("`'\"")
        if item_name.startswith("<") and item_name.endswith(">"):
            item_name = item_name[1:-1].strip()
        if item_name.startswith("{") and item_name.endswith("}"):
            item_name = item_name[1:-1].strip()
        return item_name


class PromptGPT(PromptLLM):
    def __init__(self,
                 llm_name: str,
                 prompts: list,
                 itemname_to_id: dict,
                 request_sleep: float = 0.0
                 ):
        PromptLLM.__init__(self, llm_name, prompts, itemname_to_id)
        self.request_sleep = request_sleep

    def prompt_model(self) -> (list, list):

        raw_responses = []
        reranked_recs = []
        for i, prompt in enumerate(self.prompts, start=1):

            if self.llm_name == "gpt-3.5-turbo-instruct":
                response = self.get_response_instructgpt(prompt)
            else:
                response = self.get_response_chatgpt(prompt)
            raw_responses.append(response)
            new_rank = PromptLLM.parse_response(response, self.itemname_to_id)
            reranked_recs.append(new_rank)

            if self.request_sleep > 0:
                time.sleep(self.request_sleep)
            print(f"{datetime.datetime.now()} -- Done {i}/{len(self.prompts)} users!")
        return raw_responses, reranked_recs

    def get_response_chatgpt(self,
                             prompt: str
                             ) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.llm_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            output = response.choices[0].message.content
        except Exception as e:
            try:
                import openai
                response = openai.ChatCompletion.create(
                    model=self.llm_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                output = response.choices[0].message["content"]
            except Exception as legacy_e:
                output = f"{type(e).__name__}: {e}; legacy_openai_error={legacy_e}"

        return output

    def get_response_instructgpt(self,
                                 prompt: str
                                 ) -> str:
        # model list can be found above
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.completions.create(
                model=self.llm_name,
                prompt=prompt,
                max_tokens=600,
                temperature=0,
            )
            output = response.choices[0].text
        except Exception as e:
            try:
                import openai
                response = openai.Completion.create(
                    model=self.llm_name,
                    prompt=prompt,
                    max_tokens=600,
                    temperature=0,
                )
                output = response.choices[0]["text"]
            except Exception as legacy_e:
                output = f"{type(e).__name__}: {e}; legacy_openai_error={legacy_e}"

        return output


class PromptLlama2(PromptLLM):
    def __init__(self,
                 llm_name: str,
                 prompts: list,
                 itemname_to_id: dict,
                 tokenizer_path: str,
                 model_path: str,
                 auth_token: str,
                 hf_model_id: str = None,
                 hf_cache_dir: str = None,
                 local_files_only: bool = False
                 ):
        PromptLLM.__init__(self, llm_name, prompts, itemname_to_id)
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

        self.model_id = self.resolve_model_id(llm_name, model_path, hf_model_id)
        token, token_source = self.resolve_hf_token(auth_token)
        if token:
            os.environ["HF_TOKEN"] = token
            print(f"{datetime.datetime.now()} -- Using HuggingFace token from {token_source}.")
        else:
            print(f"{datetime.datetime.now()} -- No HuggingFace token found; gated models may fail.")

        load_kwargs = {
            "token": token,
            "cache_dir": hf_cache_dir,
            "local_files_only": local_files_only,
        }
        load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **load_kwargs)
        except Exception as e:
            self.raise_hf_access_error(e)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"{datetime.datetime.now()} -- Loaded tokenizer from {self.model_id}!")

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype="auto",
                **load_kwargs
            )
        except Exception as e:
            self.raise_hf_access_error(e)
        print(f"{datetime.datetime.now()} -- Loaded model from {self.model_id}!")

        self.generation_pipe = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task='text-generation',
            do_sample=False,
            max_new_tokens=512,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    @staticmethod
    def resolve_model_id(llm_name: str, model_path: str = None, hf_model_id: str = None) -> str:
        if hf_model_id:
            return hf_model_id
        if llm_name in HF_MODEL_REPOS:
            return HF_MODEL_REPOS[llm_name]
        if "/" in llm_name:
            return llm_name
        if model_path:
            return os.path.join(model_path, llm_name)
        return llm_name

    @staticmethod
    def resolve_hf_token(auth_token: str = None) -> tuple:
        if auth_token:
            return auth_token, "--hf_auth_token"

        for env_name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"]:
            token = os.environ.get(env_name)
            if token:
                return token, env_name

        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            if token:
                return token, "Kaggle Secret HF_TOKEN"
        except Exception:
            pass

        return None, None

    def raise_hf_access_error(self, error: Exception) -> None:
        msg = str(error)
        if "gated repo" in msg.lower() or "401" in msg or "Unauthorized" in msg:
            raise RuntimeError(
                "\nCannot load gated HuggingFace model "
                f"'{self.model_id}'.\n"
                "Fix:\n"
                "1. Open the model page and request/accept access with your HuggingFace account.\n"
                "2. Create a HuggingFace access token from that same account.\n"
                "3. On Kaggle, add it as a Secret named HF_TOKEN, or pass --hf_auth_token.\n"
                "4. Enable internet for the Kaggle notebook/session.\n"
            ) from error
        raise error

    def prompt_model(self) -> (list, list):

        raw_responses = []
        reranked_recs = []

        def data_iterator():
            for prompt in iter(self.prompts):
                yield prompt

        i = 1
        for out in self.generation_pipe(data_iterator()):
            response = out[0]['generated_text']
            raw_responses.append(response)
            new_rank = PromptLLM.parse_response(response, self.itemname_to_id)
            reranked_recs.append(new_rank)
            print(f"{datetime.datetime.now()} -- Done {i}/{len(self.prompts)} users!")
            i += 1
        return raw_responses, reranked_recs


def load_helper_dicts(data_folder: str
                      ) -> (dict, dict):

    data_folder = data_folder.rstrip("/\\") + os.sep
    filename = f"{data_folder}itemid_to_name.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_name = pickle.load(fp)
    filename = f"{data_folder}itemname_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemname_to_id = pickle.load(fp)

    filename = f"{data_folder}itemid_to_namegenres.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_namegenres = pickle.load(fp)
    filename = f"{data_folder}itemnamegenres_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemnamegenres_to_id = pickle.load(fp)

    filename = f"{data_folder}itemid_to_nameplot.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemid_to_nameplot = pickle.load(fp)
    filename = f"{data_folder}itemnameplot_to_id.pkl"
    with open(f"{filename}", 'rb') as fp:
        itemnameplot_to_id = pickle.load(fp)

    return itemid_to_name, itemname_to_id, itemid_to_namegenres, itemnamegenres_to_id, itemid_to_nameplot, itemnameplot_to_id


def convert_dataframe(recs: pd.DataFrame,
                      baseline_name: str,
                      top_n: int,
                      top_m: int
                      ) -> pd.DataFrame:

    usrs = list(recs["userid"].unique())
    b_name = baseline_name.split("-")[0]
    baseline_column = [b_name for _ in range(len(usrs))]
    top_n_column = [top_n for _ in range(len(usrs))]
    top_m_column = [top_m for _ in range(len(usrs))]
    recs_column = [recs[recs["userid"] == user]["itemid"].values for user in usrs]

    d = {"userid": usrs,
         "baseline": baseline_column,
         "top_m": top_m_column,
         "top_n": top_n_column,
         "recs": recs_column
         }

    return pd.DataFrame(data=d)


def load_prompt_template(promptpath: str,
                         prompt_id: str
                         ) -> str:

    f = open(f"{promptpath}", encoding="utf-8")
    prompt_template = json.load(f)["templates"]
    f.close()
    prompt_template = {i["id"]: i["text"] for i in prompt_template}

    return prompt_template[prompt_id]


def build_prompts(recs: pd.DataFrame,
                  top_m: int,
                  top_n: int,
                  itemid_to_name: dict,
                  template: str,
                  domain: str,
                  delimiters: dict
                  ) -> list:

    # add the output format to the template
    template_prompt = template
    template_prompt = template_prompt.replace("<top_m>", str(top_m))
    template_prompt = template_prompt.replace("<top_n>", str(top_n))

    out_format_str = ""
    for i in range(1, top_n + 1):
        # out_format_str += f"{delimiters[0]}{domain} name{delimiters[1]}\n"
        out_format_str += f"{i}-> {delimiters[0]}{domain} name{delimiters[1]}\n"
    template_prompt = template_prompt.replace("<out_format_str>", out_format_str)

    users = recs.userid.values
    prompts = []
    for user in users:
        user_template = template_prompt
        # select recommendations
        user_recs = recs[recs["userid"] == user]["recs"].values[0]

        # add the baseline recommendations
        items_str = "```\n"
        for i, item in enumerate(user_recs, start=1):
            items_str += f"{i}. {itemid_to_name[item]}\n"
        items_str += "```"
        user_template = user_template.replace("<candidate_str>", items_str)
        prompts.append(user_template)

    return prompts


def main(args):

    # load helper dictionaries
    (itemid_to_name, itemname_to_id, itemid_to_namegenres,  itemnamegenres_to_id,
     itemid_to_nameplots, itemnameplot_to_id) = load_helper_dicts(args.datasetpath)

    if args.prompt_id in ["5", "6", "11", "12", "51", "52"]:  # the name of the items are augmented with genres
        itemid_to_name = itemid_to_namegenres
        itemname_to_id = itemnamegenres_to_id
    if args.prompt_id in ["21", "22", "31", "32"]:
        itemid_to_name = itemid_to_nameplots

    print(f"{datetime.datetime.now()} -- Helpers loaded!")

    # load prompt template (from json)
    prompt_template = load_prompt_template(args.promptpath, args.prompt_id)
    print(f"{datetime.datetime.now()} -- Prompt template loaded!")

    # load baseline recommendations
    recs = pd.read_csv(f"{args.datasetpath}/recs/baselines/{args.baseline_recs}", sep="\t",
                       names=["userid", "itemid", "rating"])
    recs.drop(labels="rating", axis=1, inplace=True)

    if args.run_with_sample_users:
        # load test users
        test_users = pd.read_csv(f"{args.datasetpath}/fold_{args.fold}/sample_test_users.csv",
                                 sep="\t", names=["userid"])
        recs = recs[recs["userid"].isin(test_users["userid"].values)].copy()

    print(f"{datetime.datetime.now()} -- Baseline recommendations loaded!")

    if args.debug_mode:
        debug_usrs = recs["userid"].unique()[:10]
        recs = recs[recs["userid"].isin(debug_usrs)].copy()

    # trim recommendations
    recs = recs.groupby('userid').head(args.rerank_top_m).reset_index(drop=True)
    recs = convert_dataframe(recs, args.baseline_recs, args.top_n, args.rerank_top_m)
    print(f"{datetime.datetime.now()} -- Recommendations trimmed!")

    # craft prompts from template and top-m baseline recs
    prompts = build_prompts(recs, args.rerank_top_m,  args.top_n, itemid_to_name,
                            prompt_template, args.domain, DELIMITERS.get(args.model, DEFAULT_DELIMITERS))
    recs["prompt"] = prompts
    print(f"{datetime.datetime.now()} -- Prompts crafted!")

    # calculate the number of total tokens of the prompts
    n_tokens = 0
    for p in prompts:
        n_tokens += len(p)/4
    print(f"{datetime.datetime.now()} -- Total # of tokens: {n_tokens}!")

    if "gpt" in args.model:  # prompt gpt
        if args.openai_key:
            os.environ["OPENAI_API_KEY"] = args.openai_key
        prompter = PromptGPT(args.model, prompts, itemname_to_id, args.request_sleep)
        # raw_gpt_outputs, reranked_recs = prompter.prompt_model()
    else:  # prompt llama2
        prompter = PromptLlama2(args.model, prompts, itemname_to_id,
                                args.tokenizer_path, args.model_path, args.hf_auth_token,
                                args.hf_model_id, args.hf_cache_dir, args.local_files_only)
    raw_gpt_outputs, reranked_recs = prompter.prompt_model()
    recs["raw_gpt_outputs"] = raw_gpt_outputs
    recs["reranked_recs"] = reranked_recs
    print(f"{datetime.datetime.now()} -- Done with {args.model}!")

    # save it into file
    output_name = MODEL_DICT.get(args.model, args.model.replace("/", "_"))
    out_dir = os.path.join(args.datasetpath, "recs", "reranked")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.join(out_dir, f"{output_name}-div-p{args.prompt_id}-{args.baseline_recs}.json")
    recs.to_json(out_name, orient="records")
    print(f"{datetime.datetime.now()} -- END!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetpath",
        default=False,
        type=str,
        required=True,
        help="The path to the dataset"
    )
    parser.add_argument(
        "--domain",
        default=False,
        type=str,
        required=True,
        help="The domain of recommendation (item type)"
    )
    parser.add_argument(
        "--fold",
        default=False,
        type=str,
        required=True,
        help="The data fold."
    )
    parser.add_argument(
        "--model",
        default=False,
        type=str,
        required=True,
        help="The model to use (either gpt-based or llama-based)"
    )
    parser.add_argument(
        "--model_path",
        default=False,
        type=str,
        required=False,
        help="The local path to the model (only for Llama2)"
    )
    parser.add_argument(
        "--tokenizer_path",
        default=False,
        type=str,
        required=False,
        help="The tokenizer path"
    )
    parser.add_argument(
        "--promptpath",
        default=False,
        type=str,
        required=True,
        help="The path to the prompt"
    )
    parser.add_argument(
        "--prompt_id",
        default=False,
        type=str,
        required=True,
        help="The id of the prompt"
    )
    parser.add_argument(
        "--baseline_recs",
        default=False,
        type=str,
        required=True,
        help="The baseline recommendations"
    )
    parser.add_argument(
        "--rerank_top_m",
        default=False,
        type=int,
        required=True,
        help="The # of recommendations to rerank (max 100)"
    )
    parser.add_argument(
        "--top_n",
        default=False,
        type=int,
        required=True,
        help="The # of final recommendations to provide (max 100)"
    )
    parser.add_argument(
        "--openai_key",
        default=None,
        type=str,
        required=False,
        help="The API KEY for openai"
    )
    parser.add_argument(
        "--hf_auth_token",
        default=None,
        type=str,
        required=False,
        help="The HuggingFace auth token"
    )
    parser.add_argument(
        "--hf_model_id",
        default=None,
        type=str,
        required=False,
        help="HuggingFace model id to load, e.g. meta-llama/Llama-2-7b-chat-hf"
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        type=str,
        required=False,
        help="Optional HuggingFace cache directory"
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="Load the HuggingFace model only from local cache/path"
    )
    parser.add_argument(
        "--run_with_sample_users",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script with a sample of the test users."
    )
    parser.add_argument(
        "--debug_mode",
        default=None,
        type=int,
        required=True,
        help="Whether to run the script in debug mode. The script will run with a reduced dataset size."
    )
    parser.add_argument(
        "--request_sleep",
        default=0.0,
        type=float,
        required=False,
        help="Seconds to sleep between API requests."
    )
    main(parser.parse_args())
