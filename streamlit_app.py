import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ==========================================
# تنظیمات اولیه
# ==========================================
load_dotenv()  # کلید را از فایل .env می‌خواند (برای اجرای محلی روی لپ‌تاپ)

api_key = os.environ.get("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.6-flash"  # اگر گوگل دوباره این مدل را عوض کرد، فقط همین خط را به‌روز کن

st.set_page_config(page_title="دستیار محتوای اینستاگرام", page_icon="📸", layout="centered")
st.title("📸 دستیار تولید محتوای اینستاگرام")
st.caption("یک عکس از محصول بکش و رها کن و/یا توضیحاتت را بنویس تا هوش مصنوعی برایت محتوا بسازد.")

if not api_key:
    st.error(
        "کلید GEMINI_API_KEY پیدا نشد. اگر روی Streamlit Cloud هستی، از منوی Settings > Secrets "
        "این خط را اضافه کن:\nGEMINI_API_KEY = \"کلید_شما\"\n\n"
        "اگر روی لپ‌تاپ خودت اجرا می‌کنی، یک فایل به اسم .env بساز و همین خط را داخلش بنویس."
    )
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# ورودی‌های کاربر
# ==========================================
uploaded_image = st.file_uploader(
    "عکس محصول را اینجا بکش و رها کن (یا کلیک کن)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded_image is not None:
    st.image(uploaded_image, caption="پیش‌نمایش عکس", width="stretch")

user_text = st.text_area(
    "توضیحات یا ایده‌های خودت (اختیاری، ولی هرچه کامل‌تر باشد بهتر است)",
    height=150,
    placeholder="مثلاً: نام برند، ویژگی خاصی که در عکس معلوم نیست، هدف از پست، تخفیف ویژه و ...",
)

generate_btn = st.button("🚀 تولید محتوا", type="primary")


# ==========================================
# تابع کمکی: یک تماس ساده و سبک با Gemini
# ==========================================
def ask_gemini(system_role: str, instruction: str, raw_input: str) -> str:
    prompt = (
        f"نقش تو: {system_role}\n\n"
        f"دستور: {instruction}\n\n"
        f"اطلاعات خام محصول:\n{raw_input}\n\n"
        "خروجی را فقط به زبان فارسی و آماده برای استفاده مستقیم بنویس، بدون مقدمه‌چینی اضافه."
    )
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


# ==========================================
# تحلیل عکس با Gemini Vision
# ==========================================
def analyze_image(image_bytes: bytes, mime_type: str) -> str:
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
# تولید محتوا با ۴ نقش پشت سر هم (بدون crewai، سبک و سریع)
# هر بخش بلافاصله بعد از آماده شدن نمایش داده می‌شود تا ارتباط
# با مرورگر طولانی و بی‌صدا نماند (که باعث قطع ارتباط می‌شود)
# ==========================================
def run_content_pipeline(raw_input: str) -> str:
    status = st.empty()

    status.info("در حال نوشتن بخش استوری...")
    story = ask_gemini(
        "متخصص تولید محتوای استوری اینستاگرام، یک ادمین حرفه‌ای که می‌داند چطور مخاطب را در استوری درگیر کند",
        "۳ ایده متنی کوتاه، صمیمی و تعاملی برای استوری بنویس (شامل نظرسنجی و کال‌تو‌اکشن).",
        raw_input,
    )
    st.markdown("#### 📱 بخش استوری")
    st.markdown(story)

    status.info("در حال نوشتن بخش ریلز...")
    reels = ask_gemini(
        "سناریونویس ریلز اینستاگرام، کارگردان و کپی‌رایتر شبکه‌های اجتماعی",
        "یک سناریوی ۱۵ ثانیه‌ای برای ریلز بنویس. ۳ ثانیه اول باید قلاب (Hook) قوی داشته باشد، به همراه کپشن.",
        raw_input,
    )
    st.markdown("#### 🎬 بخش ریلز")
    st.markdown(reels)

    status.info("در حال نوشتن بخش اسلایدی...")
    carousel = ask_gemini(
        "طراح پست‌های اسلایدی، استراتژیست محتوا",
        "این اطلاعات را به یک پست ۵ اسلایدی با تیترهای جذاب تبدیل کن.",
        raw_input,
    )
    st.markdown("#### 🖼️ بخش اسلایدی")
    st.markdown(carousel)

    combined_draft = (
        f"### بخش استوری\n{story}\n\n"
        f"### بخش ریلز\n{reels}\n\n"
        f"### بخش اسلایدی\n{carousel}\n"
    )

    status.info("در حال بازبینی نهایی و تمیزکاری متن...")
    final_result = ask_gemini(
        "مدیر کنترل کیفیت و سردبیر سخت‌گیر، وظیفه‌ات تحویل یک خروجی بی‌نقص است",
        (
            "متن زیر که توسط سه نویسنده مختلف (استوری، ریلز، اسلایدی) نوشته شده را بخوان، "
            "خطاهای نگارشی را اصلاح کن، لحن را جذاب‌تر کن و خروجی نهایی را دقیقاً با همین سه بخش "
            "(استوری، ریلز، اسلایدی) و همین ترتیب، تمیز و دسته‌بندی‌شده تحویل بده."
        ),
        combined_draft,
    )
    status.empty()

    return final_result


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
            with st.spinner("در حال تولید محتوا... چند لحظه صبر کن."):
                try:
                    final_result = run_content_pipeline(combined_input)
                    st.subheader("✅ خروجی نهایی")
                    st.markdown(final_result)
                    st.download_button(
                        "📥 دانلود به‌صورت فایل متنی",
                        data=final_result,
                        file_name="instagram_content.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"خطا در تولید محتوا: {e}")
