import streamlit as st
import boto3
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

logo = Image.open("logo.png")
st.set_page_config(page_title="DevHomey AWS Assistant", layout="wide", page_icon=logo)

col1, col2 = st.columns([1, 6])
with col1:
    st.image(logo, width=180)
with col2:
    st.title("🧠 DevHomey: AWS Billing + Live Resource Analyzer")

tab1, tab2, tab3 = st.tabs([
    "📄 PDF Billing Upload",
    "☁️ Live AWS Resource Data",
    "⚙️ EC2 DevOps Actions"
])

# Tab 1: PDF Billing Upload
with tab1:
    uploaded_file = st.file_uploader("📎 Upload AWS Billing PDF", type="pdf")

    if uploaded_file:
        with open("temp_uploaded.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp_uploaded.pdf")
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])

        matches = re.findall(r"([A-Za-z ]+?)\s+USD(?:\xa0|\s)?([0-9,]+\.\d{2})", full_text)
        data = []

        for name, amount in matches:
            if name.lower() not in ["total", "grand total", "amazon web services india private limited"]:
                data.append({"Service": name.strip(), "Cost": float(amount.replace(",", ""))})

        if data:
            df = pd.DataFrame(data)
            df = df.groupby("Service", as_index=False).sum().sort_values(by="Cost", ascending=False)

            st.subheader("💰 Top 3 Costly Services")
            st.table(df.head(3))

            fig, ax = plt.subplots()
            top_services = df.head(5)
            ax.pie(top_services["Cost"], labels=top_services["Service"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")
            st.pyplot(fig)

            embeddings = OpenAIEmbeddings()
            db = FAISS.from_documents(pages, embeddings)
            db.save_local("pdf_vector_store")

            st.subheader("💬 Ask Questions from Your PDF")
            question = st.text_input("Type your question:")
            if question:
                chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                    retriever=db.as_retriever(),
                    return_source_documents=False
                )
                answer = chain.run(question)
                st.success(answer)
        else:
            st.error("❌ No valid AWS service cost entries found in the PDF.")
    else:
        st.info("📄 Please upload a PDF to get started.")

# Tab 2: Live AWS Resource Data
with tab2:
    st.subheader("🔍 Fetching Live EC2 Instances and S3 Buckets")

    try:
        ec2 = boto3.client("ec2", region_name="ap-south-1")
        s3 = boto3.client("s3", region_name="ap-south-1")

        ec2_data = ec2.describe_instances()
        instances = []
        for r in ec2_data["Reservations"]:
            for inst in r["Instances"]:
                instances.append({
                    "InstanceId": inst["InstanceId"],
                    "Type": inst["InstanceType"],
                    "State": inst["State"]["Name"],
                    "AZ": inst["Placement"]["AvailabilityZone"]
                })

        if instances:
            st.write("✅ **EC2 Instances**")
            st.dataframe(pd.DataFrame(instances))
        else:
            st.info("No EC2 instances found.")

        buckets = s3.list_buckets()
        s3_data = [{"Bucket Name": b["Name"], "Creation Date": b["CreationDate"].strftime("%Y-%m-%d")} for b in buckets.get("Buckets", [])]

        if s3_data:
            st.write("✅ **S3 Buckets**")
            st.dataframe(pd.DataFrame(s3_data))
        else:
            st.info("No S3 buckets found.")

    except Exception as e:
        st.error(f"❌ Error fetching AWS data: {e}")

# Tab 3: EC2 DevOps Actions
with tab3:
    st.subheader("⚙️ EC2 DevOps Actions")

    ec2 = boto3.client("ec2", region_name="ap-south-1")
    ec2_data = ec2.describe_instances()
    instances = []
    instance_ids = []

    for r in ec2_data["Reservations"]:
        for inst in r["Instances"]:
            instance_ids.append(inst["InstanceId"])
            instances.append({
                "InstanceId": inst["InstanceId"],
                "Type": inst["InstanceType"],
                "State": inst["State"]["Name"],
                "AZ": inst["Placement"]["AvailabilityZone"]
            })

    if instances:
        selected_instance = st.selectbox("🖥️ Select an EC2 instance to manage", instance_ids)

        action = st.radio("Select action", ["Start", "Stop", "Reboot", "Terminate"])
        if st.button("🔧 Execute"):
            try:
                if action == "Start":
                    ec2.start_instances(InstanceIds=[selected_instance])
                elif action == "Stop":
                    ec2.stop_instances(InstanceIds=[selected_instance])
                elif action == "Reboot":
                    ec2.reboot_instances(InstanceIds=[selected_instance])
                elif action == "Terminate":
                    ec2.terminate_instances(InstanceIds=[selected_instance])
                st.success(f"{action} command issued for {selected_instance}")
            except Exception as e:
                st.error(f"Error performing {action}: {e}")
    else:
        st.warning("No EC2 instances found.")

    st.markdown("---")
    st.subheader("🚀 Launch New EC2 Instance")

    ami = st.text_input("AMI ID", value="ami-0cda377a1b884a1bc")
    instance_type = st.selectbox("Instance Type", ["t2.micro", "t2.small", "t3.micro"])
    key_name = st.text_input("Key Pair Name (must exist)")

    if st.button("🆕 Launch Instance"):
        try:
            response = ec2.run_instances(
                ImageId=ami,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=key_name
            )
            instance_id = response['Instances'][0]['InstanceId']
            st.success(f"Launched instance {instance_id}")
        except Exception as e:
            st.error(f"Error launching instance: {e}")

    st.markdown("---")
    st.subheader("🔄 Migrate Instance (Snapshot → New VM)")
    st.markdown("This will create a snapshot of the selected instance and launch a new instance with the same root volume.")

    migrate_instance = st.selectbox("Select Instance to Migrate", instance_ids)

    if st.button("📦 Migrate"):
        try:
            vols = ec2.describe_instances(InstanceIds=[migrate_instance])["Reservations"][0]["Instances"][0]["BlockDeviceMappings"]
            root_vol_id = vols[0]["Ebs"]["VolumeId"]

            snapshot = ec2.create_snapshot(VolumeId=root_vol_id, Description="Snapshot for migration")
            snapshot_id = snapshot["SnapshotId"]

            waiter = ec2.get_waiter("snapshot_completed")
            st.info("⏳ Waiting for snapshot to complete...")
            waiter.wait(SnapshotIds=[snapshot_id])

            block_device = [{
                "DeviceName": vols[0]["DeviceName"],
                "Ebs": {
                    "SnapshotId": snapshot_id,
                    "DeleteOnTermination": True,
                    "VolumeType": "gp2"
                }
            }]

            response = ec2.run_instances(
                ImageId=ami,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=key_name,
                BlockDeviceMappings=block_device
            )
            new_instance_id = response['Instances'][0]['InstanceId']
            st.success(f"Migrated instance launched: {new_instance_id}")
        except Exception as e:
            st.error(f"Migration failed: {e}")
