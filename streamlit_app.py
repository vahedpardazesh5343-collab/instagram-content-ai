import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from crewai import Agent, Task, Crew, Process, LLM

# ==========================================
# تنظیمات اولیه
# ==========================================
load_dotenv()  # کلید را از فایل .env می‌خواند

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        api_key = None

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key  # تا crewai/LLM هم بتواند آن را پیدا کند

MODEL_NAME = "gemini-3.6-flash"  # اگر گوگل دوباره این مدل را عوض کرد، فقط همین خط را به‌روز کن

st.set_page_config(page_title="دستیار محتوای اینستاگرام", page_icon="📸", layout="centered")
st.title("📸 دستیار تولید محتوای اینستاگرام")
st.caption("یک عکس از محصول بکش و رها کن و/یا توضیحاتت را بنویس تا تیم هوش مصنوعی برایت محتوا بسازد.")

if not api_key:
    st.error(
        "کلید GEMINI_API_KEY پیدا نشد. یک فایل به اسم .env در همین پوشه بساز و "
        "این خط را داخلش بنویس:\nGEMINI_API_KEY=کلید_شما"
    )
    st.stop()

# ==========================================
# ورودی‌های کاربر
# ==========================================
uploaded_image = st.file_uploader(
    "عکس محصول را اینجا بکش و رها کن (یا کلیک کن)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="پیش‌نمایش عکس", use_container_width=True)

user_text = st.text_area(
    "توضیحات یا ایده‌های خودت (اختیاری، ولی هرچه کامل‌تر باشد بهتر است)",
    height=150,
    placeholder="مثلاً: نام برند، ویژگی خاصی که در عکس معلوم نیست، هدف از پست، تخفیف ویژه و ...",
)

generate_btn = st.button("🚀 تولید محتوا", type="primary")


# ==========================================
# تحلیل عکس با Gemini Vision
# ==========================================
def analyze_image(image_bytes: bytes, mime_type: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
            (
                "این عکس یک محصول برای فروش است. جنس، رنگ، کیفیت ظاهری، جزئیات دوخت/طراحی "
                "و هر نکته‌ای که برای تبلیغات اینستاگرامی مفید است را توصیف کن. "
                "خروجی را فقط به زبان فارسی و در چند خط کوتاه و کاربردی بنویس، بدون مقدمه‌چینی."
            ),
        ],
    )
    return response.text


# ==========================================
# تعریف و اجرای تیم ایجنت‌ها (همان منطق قبلی)
# ==========================================
def run_crew(raw_input: str) -> str:
    my_llm = LLM(model=f"gemini/{MODEL_NAME}", api_key=api_key)

    # به‌جای ۳ ایجنت جدا (استوری/ریلز/اسلایدی) که هر کدام یک درخواست API جدا مصرف می‌کردند،
    # این ۳ نقش را در یک ایجنت واحد ادغام کرده‌ایم تا مصرف سهمیه‌ی روزانه‌ی Gemini کمتر شود.
    content_agent = Agent(
        role="استراتژیست محتوای اینستاگرام (استوری + ریلز + اسلایدی)",
        goal="تولید هم‌زمان محتوای استوری، ریلز و پست اسلایدی از روی یک ورودی خام، هرکدام با کیفیت تخصصی خودش.",
        backstory=(
            "تو یک تیم یک‌نفره‌ی حرفه‌ای اینستاگرام هستی که هم‌زمان تسلط کامل روی نوشتن استوری صمیمی، "
            "سناریوی پرقلاب ریلز، و طراحی پست اسلایدی آموزشی داری."
        ),
        allow_delegation=False,
        llm=my_llm,
        max_retry_limit=1,
    )
    qa_agent = Agent(
        role="مدیر کنترل کیفیت و بازرس نهایی",
        goal="بررسی دقیق محتوای تولید شده، اصلاح خطاهای نگارشی و تایید نهایی.",
        backstory="تو یک سردبیر سخت‌گیر هستی. وظیفه تو این است که یک خروجی بی‌نقص تحویل دهی.",
        allow_delegation=False,
        llm=my_llm,
        max_retry_limit=1,
    )

    task_content = Task(
        description=(
            "با استفاده از این اطلاعات محصول، سه بخش کاملاً جدا و کامل بنویس:\n"
            "۱) استوری: ۳ ایده متنی کوتاه برای استوری اینستاگرام (شامل نظرسنجی و کال‌تو‌اکشن)\n"
            "۲) ریلز: یک سناریوی ۱۵ ثانیه‌ای با قلاب قوی در ۳ ثانیه اول، به‌همراه کپشن\n"
            "۳) اسلایدی: تبدیل اطلاعات به یک پست ۵ اسلایدی با تیترهای جذاب\n\n"
            f"اطلاعات محصول:\n{raw_input}\n\n"
            "هر بخش را با یک عنوان مشخص (مثلاً «### استوری») جدا کن تا کاملاً از هم متمایز باشند."
        ),
        expected_output="سه بخش کامل و جداگانه: استوری، ریلز، و پست اسلایدی.",
        agent=content_agent,
    )
    task_qa_review = Task(
        description="محتوای تولید‌شده را بخوان، لحن را جذاب‌تر کن و خروجی نهایی را به‌صورت دسته‌بندی‌شده تحویل بده.",
        expected_output="متن تمیز شامل: بخش استوری، ریلز و اسلایدی.",
        agent=qa_agent,
        context=[task_content],
    )

    crew = Crew(
        agents=[content_agent, qa_agent],
        tasks=[task_content, task_qa_review],
        process=Process.sequential,
        verbose=False,
    )
    result = crew.kickoff()
    return str(result)


# ==========================================
# اجرای اصلی
# ==========================================
if generate_btn:
    if not uploaded_image and not user_text.strip():
        st.warning("حداقل یک عکس آپلود کن یا یک توضیح بنویس.")
    else:
        combined_parts = []

        if uploaded_image is not None:
            with st.spinner("در حال تحلیل عکس با هوش مصنوعی..."):
                try:
                    image_bytes = uploaded_image.getvalue()
                    mime_type = uploaded_image.type or "image/jpeg"
                    image_description = analyze_image(image_bytes, mime_type)
                    st.subheader("🔍 تحلیل عکس توسط هوش مصنوعی")
                    st.info(image_description)
                    combined_parts.append(
                        f"توضیحات ظاهری محصول (استخراج‌شده از عکس):\n{image_description}"
                    )
                except Exception as e:
                    st.error(f"خطا در تحلیل عکس: {e}")

        if user_text.strip():
            combined_parts.append(f"توضیحات و ایده‌های اضافه از طرف کاربر:\n{user_text.strip()}")

        combined_input = "\n\n".join(combined_parts)

        if combined_input:
            with st.spinner("تیم ایجنت‌ها در حال تولید محتوا هستند... ممکن است چند دقیقه طول بکشد."):
                try:
                    final_result = run_crew(combined_input)
                    st.subheader("✅ خروجی نهایی")
                    st.markdown(final_result)
                    st.download_button(
                        "📥 دانلود به‌صورت فایل متنی",
                        data=final_result,
                        file_name="instagram_content.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"خطا در اجرای تیم ایجنت‌ها: {e}")
