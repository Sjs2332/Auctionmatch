import { NextResponse } from 'next/server';

export async function POST(request: Request) {
    try {
        const formData = await request.formData();

        // Proxy the request to FastAPI
        const response = await fetch('http://localhost:8000/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Upload error:', error);
        return NextResponse.json({ error: 'Failed to upload file' }, { status: 500 });
    }
}
