import os
import sys
from contextlib import redirect_stdout

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class ProductInfo(BaseModel):
    name: str = Field(description="Name of the product")
    price: float = Field(description="Price in INR")


def run_demo():
    print("===== LangChain Output Parser Output =====\n")

    print("1. StrOutputParser")
    prompt = PromptTemplate.from_template("Translate to French: {text}")
    rendered_prompt = prompt.format(text="Good morning")
    raw_model_output = "Bonjour"
    parsed_output = StrOutputParser().parse(raw_model_output)
    print("Prompt:", rendered_prompt)
    print("Raw model output:", raw_model_output)
    print("Parsed output:", parsed_output)

    print("\n2. CommaSeparatedListOutputParser")
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate.from_template(
        "List 5 programming languages, comma-separated."
    )
    rendered_prompt = prompt.format()
    raw_model_output = "Python, Java, C++, JavaScript, Ruby"
    parsed_output = parser.parse(raw_model_output)
    print("Prompt:", rendered_prompt)
    print("Raw model output:", raw_model_output)
    print("Parsed output:", parsed_output)

    print("\n3. PydanticOutputParser")
    parser = PydanticOutputParser(pydantic_object=ProductInfo)
    prompt = PromptTemplate(
        template=(
            "Extract product name and price from: {text}\n"
            "{format_instructions}"
        ),
        input_variables=["text"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
        },
    )
    rendered_prompt = prompt.format(
        text="The Redmi Note 12 is available for INR 14999."
    )
    raw_model_output = '{"name": "Redmi Note 12", "price": 14999.0}'
    parsed_output = parser.parse(raw_model_output)
    print("Prompt preview:", rendered_prompt.splitlines()[0])
    print("Raw model output:", raw_model_output)
    print("Parsed output:", parsed_output)
    print("Product name:", parsed_output.name)
    print("Product price:", parsed_output.price)

    print("\n4. JsonOutputParser")
    parser = JsonOutputParser()
    prompt = PromptTemplate(
        template=(
            "Extract company and founder from the text: {text}\n"
            "Return valid JSON with keys company and founder."
        ),
        input_variables=["text"],
    )
    rendered_prompt = prompt.format(
        text="Hope AI was founded by Ramisha Rani in Tamil Nadu."
    )
    raw_model_output = '{"company": "Hope AI", "founder": "Ramisha Rani"}'
    parsed_output = parser.parse(raw_model_output)
    print("Prompt:", rendered_prompt)
    print("Raw model output:", raw_model_output)
    print("Parsed output:", parsed_output)
    print("Company:", parsed_output["company"])
    print("Founder:", parsed_output["founder"])


def main():
    output_path = os.path.join(os.path.dirname(__file__), "Output16.txt")

    with open(output_path, "w", encoding="utf-8") as output_file:
        with redirect_stdout(Tee(sys.stdout, output_file)):
            run_demo()


if __name__ == "__main__":
    main()
