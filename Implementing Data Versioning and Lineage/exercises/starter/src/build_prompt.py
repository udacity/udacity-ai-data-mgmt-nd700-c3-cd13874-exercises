from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/inventory.csv")
RULES_PATH = Path("prompts/base_rules.txt")
OUTPUT_PATH = Path("artifacts/system_prompt.txt")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    rules = RULES_PATH.read_text(encoding="utf-8").strip()

    summary_lines = []
    summary_lines.append("Current supermarket inventory snapshot:")
    for _, row in df.iterrows():
        low_stock = row["store_qty"] < row["reorder_point"]
        stock_flag = "LOW_STOCK" if low_stock else "OK"
        summary_lines.append(
            f"- {row['product_name']} | "
            f"category={row['category']} | "
            f"brand={row['brand']} | "
            f"store_qty={row['store_qty']} | "
            f"warehouse_qty={row['warehouse_qty']} | "
            f"reorder_point={row['reorder_point']} | "
            f"supplier={row['supplier']} | "
            f"lead_time_days={row['lead_time_days']} | "
            f"aisle={row['aisle']} | "
            f"status={stock_flag}"
        )

    inventory_summary = "\n".join(summary_lines)

    prompt = f"""
{rules}

Operational context:
- The assistant supports store and supply-chain questions for a supermarket.
- Use retrieved documents as the source of truth.
- Do not invent products or quantities.

Inventory snapshot summary:
{inventory_summary}
""".strip()

    OUTPUT_PATH.write_text(prompt, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()