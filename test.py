import pandas as pd
import random
import re

# 1. 模拟东莞真实企业库生成器
towns = ['长安镇', '塘厦镇', '常平镇', '厚街镇', '凤岗镇', '虎门镇', '黄江镇', '寮步镇', '大朗镇', '清溪镇', '大岭山镇',
         '麻涌镇']
brand_names = ['泰森', '创维', '精诚', '宏达', '领航', '捷丰', '鑫原', '鼎泰', '科迈', '恒远', '正大', '华盈', '信义',
               '博拓']
industry_suffix = ['精密模具', '五金电子', '机械设备', '自动化科技', '塑胶制品', '精机', '智能装备', '模具配件']


def generate_company_name():
    return f"东莞市{random.choice(towns)}{random.choice(brand_names)}{random.choice(industry_suffix)}有限公司"


# 2. 字段清洗函数（保留数字和mm）
def clean_unit_field(text, is_max=True):
    if not text: return ""
    text = str(text).lower().replace(' ', '')
    # 提取数字部分并统一带上 mm
    nums = re.findall(r"\d+\.?\d*", text)
    if is_max:
        # K列格式：数字mm*数字mm*数字mm
        return "*".join([f"{n}" for n in nums[:3]])
    else:
        # L列格式：数字mm
        return f"{nums[0]}" if nums else ""


# 3. 构建 30,000 条数据
def create_30k_data():
    rows = []
    industries = ['通用设备制造业', '金属制品业', '橡胶和塑料制品业', '计算机、通信和其他电子设备制造业']
    certs = ['ISO9001', 'ISO9001、ISO14001', 'ISO9001、TS16949']
    honor = ['高新技术企业', '专精特新企业', '科技型中小企业', '倍增企业', '国家级专精特新小巨人']

    for i in range(1, 30001):
        town = random.choice(towns)
        # 随机生成尺寸逻辑
        k_val = f"{random.randint(500, 5000)}*{random.randint(400, 3000)}*{random.randint(300, 2000)}"
        l_val = f"{random.uniform(0.01, 0.5):.2f}"

        row = {
            "序号": i,
            "企业名称": generate_company_name(),
            "统一社会信用代码": f"91441900MA{random.randint(10000, 99999)}{random.choice('ABCDEFGHJKL')}",
            "企业类型": "有限责任公司 (自然人投资或控股)",
            "最新存续变更时间": f"2025-{random.randint(5, 12):02d}-{random.randint(1, 28):02d}",
            "所属行业": random.choice(industries),
            "质量体系认证": random.choice(certs),
            "其他资质认证": random.choice(honor),
            "主营产品 / 产品案例": f"{random.choice(['精密零件', '注塑模具', '自动化机床', '电子连接器'])}加工",
            "核心加工设备": random.sample(['CNC加工中心', '精雕机', '海天注塑机', '大族激光', '日本沙迪克火花机','双螺杆挤出机','精密钣金设备','贴片机','SMT生产线'], random.randint(2, 3)),
            "最大加工尺寸(mm)": k_val,
            "最小加工尺寸(mm)": l_val,
            "加工精度 / 公差(mm)": f"±0.00{random.randint(1, 9)}",
            "模具最大重量 (kg)": random.randint(500, 15000),
            "年产套数 / 年产能": f"{random.randint(100, 5000)}",
            "企业注册地址": f"东莞市{town}{random.choice(['工业大道', '科技园', '中心路'])}10{i % 9}号",
            "企业联系电话": f"0769-{random.randint(81000000, 89999999)}"
        }
        rows.append(row)
    return rows


# 4. 导出并优化间距
df = pd.DataFrame(create_30k_data())
writer = pd.ExcelWriter('东莞制造业3万条.xlsx', engine='xlsxwriter')
df.to_excel(writer, index=False, sheet_name='Sheet1')

workbook = writer.book
worksheet = writer.sheets['Sheet1']

# 自动加宽每一列，方便观看
for i, col in enumerate(df.columns):
    column_len = max(df[col].astype(str).map(len).max(), len(col)) + 5  # 额外+5间距
    worksheet.set_column(i, i, column_len)

writer.close()
print("文件已生成：东莞制造业3万条.xlsx")